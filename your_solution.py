import glob
import math
import os
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    "output_dir": f"{OUTPUT_DIR}/qwen3-1b-russian",
    "optim": "adamw_torch_fused",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "gradient_accumulation_steps": 4,  # Effective Batch Size = 32 * 4 * 512 = 65,536 tokens/step
    "learning_rate": 6e-4,  # Standard peak LR for 1B pretraining from scratch
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,  # Rapid warmup for 15-min run
    "logging_steps": 5,
    "eval_steps": 50,
    "eval_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "bf16": True,
    "tf32": True,
    "gradient_checkpointing": False,  # Not needed for 512 seq_len on 80GB VRAM
    "dataloader_num_workers": 8,
    "dataloader_pin_memory": True,
    "torch_compile": False,  # Set False to avoid 2-3 min JIT compilation overhead during 15-min budget
    "report_to": "none",
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """
    TODO: Implement tokenizer preparation.
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    TODO: Implement tokenization function.
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors=None,
    )
    # маскируем падинги
    tokenized[LABELS] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in seq
        ]
        for seq in tokenized[INPUT_IDS]
    ]
    return tokenized


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    TODO: Implement saving dataset as parquet shards.
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving dataset into {num_shards} shards in {output_dir}...")
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard_path = os.path.join(output_dir, f"{i:05d}.parquet")
        shard.to_parquet(shard_path)
    print("Dataset successfully tokenized and saved as parquet files.")


def prepare_dataset():
    """
    TODO: Implement dataset preparation.
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    print("Loading raw Wikipedia dataset...")
    raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()

    print("Tokenizing dataset across multiple CPU workers...")
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=os.cpu_count() or 8,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )
    save_as_parquets(tokenized_dataset, OUTPUT_DIR, NUM_SHARDS)



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    TODO: Implement loading of tokenized dataset from parquet files.
    - List only parquet files in data_dir, sorted by filename
    - Load them using load_dataset('parquet', data_files=...)
    - Return the 'train' split
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Жаль, сегодня без паркета в {data_dir}")
    print(f"Loading {len(parquet_files)} parquet shards...")
    return load_dataset("parquet", data_files=parquet_files, split="train")


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def generate_samples(
    model, 
    tokenizer, 
    prompts=None, 
    max_new_tokens=64, 
    device="cuda"
):
    if prompts is None:
        prompts = [
            "Смешарики - это",
            "в 1989 году на площади тяньаньмэнь",
            "Один серый другой белый - два веселых",
            "Да пребудет с тобой",
        ]

    model.eval()
    print("\n" + "=" * 50)
    print("--- QUALITATIVE GENERATION CHECK ---")
    print("=" * 50)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Generation: {decoded}\n" + "-" * 30)


def train_model():
    """
    TODO: Implement the training pipeline.
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from TRAINING_CONFIG
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Save metric history to trainer_state.json for local loss plots
    """
    tokenizer = prepare_tokenizer()
    full_dataset = load_tokenized_dataset(OUTPUT_DIR)
    train_dataset, eval_dataset = split_dataset(full_dataset, VALIDATION_SIZE)

    model = create_model(tokenizer)

    training_args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
    )

    print("\n[START] Starting 15-minute pretraining session...")
    trainer.train()

    print("\n[EVAL] Running final evaluation...")
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", float("nan"))
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    print(f"Final Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.2f}")

    trainer.save_state()
    trainer.save_model(os.path.join(TRAINING_CONFIG["output_dir"], "best_model"))

    # Generate test text to evaluate grammatical & syntactic coherence
    generate_samples(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    if not os.path.exists(OUTPUT_DIR) or not glob.glob(
        os.path.join(OUTPUT_DIR, "*.parquet")
    ):
        print("Data shards not found. Preparing dataset...")
        prepare_dataset()
    
    # Step 2: Train the model
    train_model()
