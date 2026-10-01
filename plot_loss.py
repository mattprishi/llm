import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    state_file = "artifacts/trainer_state.json"
    with open(state_file, "r") as f:
        data = json.load(f)

    logs = data.get("log_history", [])
    
    steps = [x["step"] for x in logs if "loss" in x]
    losses = [x["loss"] for x in logs if "loss" in x]
    lrs = [x["learning_rate"] for x in logs if "learning_rate" in x]

    eval_steps = [x["step"] for x in logs if "eval_loss" in x]
    eval_losses = [x["eval_loss"] for x in logs if "eval_loss" in x]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Train Loss (сырой)", alpha=0.3, color="blue")
    
    if len(losses) > 5:
        smoothed = np.convolve(losses, np.ones(5)/5, mode="valid")
        plt.plot(steps[len(steps)-len(smoothed):], smoothed, label="Train Loss (сглаженный)", color="blue", lw=2)

    if eval_losses:
        plt.scatter(eval_steps, eval_losses, color="red", zorder=5, label="Eval Loss")

    plt.title("Динамне Qwen3-1B (15 минут на A100)")
    plt.xlabel("Шаг (Step)")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/loss_curve.png", dpi=300)
    print("График сохранен в artifacts/loss_curve.png")

if __name__ == "__main__":
    main()

