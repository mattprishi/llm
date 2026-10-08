import json
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics(state_path):
    if not os.path.exists(state_path):
        return [], [], [], []
    with open(state_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logs = data.get("log_history", [])
    train_steps = [x["step"] for x in logs if "loss" in x]
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_steps = [x["step"] for x in logs if "eval_loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]
    return train_steps, train_loss, eval_steps, eval_loss

def smooth(vals, window=5):
    if len(vals) < window:
        return vals
    return np.convolve(vals, np.ones(window)/window, mode="valid")

def main():
    plt.figure(figsize=(11, 5.5), dpi=300)

    # Run 1
    s1, l1, _, _ = extract_metrics("artifacts/trainer_state.json")
    if l1:
        sm1 = smooth(l1)
        plt.plot(s1[len(s1)-len(sm1):], sm1, label="Run 1 (Baseline: batch=128, lr=6e-4)", color="gray", lw=1.5, alpha=0.6)

    # Run 2
    s2, l2, es2, el2 = extract_metrics("artifacts/trainer_state_run2.json")
    if l2:
        sm2 = smooth(l2)
        plt.plot(s2[len(s2)-len(sm2):], sm2, label="Run 2 (batch=64, lr=8e-4)", color="#ff7f0e", lw=1.8, alpha=0.7)
        if el2:
            plt.scatter(es2, el2, color="#ff7f0e", marker="o", s=40)

    # Run 3
    s3, l3, es3, el3 = extract_metrics("artifacts/trainer_state_run3.json")
    if l3:
        sm3 = smooth(l3)
        plt.plot(s3[len(s3)-len(sm3):], sm3, label="Run 3 (batch=32, lr=1e-3, zero eval overhead)", color="#1f77b4", lw=2.2)
        if el3:
            plt.scatter(es3, el3, color="#1f77b4", marker="s", s=60, label="Run 3 Final Eval")

    plt.title("Сравнение сходимости претрейна Qwen3-1B за 15 минут (A100)")
    plt.xlabel("Шаги оптимизации")
    plt.ylabel("Train Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/loss_curve.png", dpi=300)
    print("График сохранен в artifacts/loss_curve.png")

if __name__ == "__main__":
    main()
