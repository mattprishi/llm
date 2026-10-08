# Текст для формы сдачи

Репозиторий: https://github.com/mattprishi/llm

На одной NVIDIA A100 80 GB сравнил восемь успешных запусков шести уникальных
конфигураций Qwen3-1B при неизменных данных, seq_len=512, BF16, AdamW, LR,
scheduler и seed. Baseline за 300 секунд показал 7 067 useful tokens/s. Лучший
финальный запуск FlashAttention 2 + sequence packing показал 22 238 useful
tokens/s, ускорение S=3.1465 и validation loss 6.6746. FlashAttention 2 +
padding-free дал 18 204 tokens/s (S=2.5758), снизил peak allocated memory с
41.24 до 15.18 GiB и достиг лучшего validation loss 5.1540. Activation
checkpointing снизил память до 17.39 GiB, но замедлил обучение на 15%. Попытки
batch=64 завершились OOM; torch.compile и Liger оказались несовместимы с Triton
на driver 470, что зафиксировано в отчёте. Также реализованы и проверены на 2 и 4
Gloo-процессах ring_reduce_scatter, ring_all_gather и ring_all_reduce.
