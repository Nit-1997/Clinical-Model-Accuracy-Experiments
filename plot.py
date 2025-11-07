import matplotlib.pyplot as plt

fields = ["age", "systolic_bp", "diastolic_bp", "heart_rate", "diagnosis", "treatment", "outcome"]
medgemma = [0.702, 0.999, 0.999, 0.857, 1.0, 0.997, 1.0]
gemma3 = [0.124, 0.135, 0.214, 0.0, 0.145, 0.002, 0.335]


def main():
    plt.figure(figsize=(9, 5))
    plt.bar(fields, medgemma, label="MedGemma-4B-IT", alpha=0.75)
    plt.bar(fields, gemma3, label="Gemma-3-270M-IT", alpha=0.75)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.ylim(0, 1.05)
    plt.title("Field-wise Accuracy Comparison", fontsize=13)
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()