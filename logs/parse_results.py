# Graphing made with the help of ChatGPT
import re
import matplotlib.pyplot as plt
import pandas as pd

log_data = ""

# Fixing the issue by adjusting regex to properly handle scientific notation with 'e+00' or 'e-00'
pattern_fixed = re.compile(
    r"Epoch (\d+)/\d+.*?"
    r"accuracy: ([\d\.e\+\-]+).*?"
    r"f1_score: ([\d\.e\+\-]+).*?"
    r"loss: ([\d\.e\+\-]+).*?"
    r"precision_13: ([\d\.e\+\-]+).*?"
    r"recall_13: ([\d\.e\+\-]+).*?"
    r"val_accuracy: ([\d\.e\+\-]+).*?"
    r"val_f1_score: ([\d\.e\+\-]+).*?"
    r"val_loss: ([\d\.e\+\-]+).*?"
    r"val_precision_13: ([\d\.e\+\-]+).*?"
    r"val_recall_13: ([\d\.e\+\-]+)", re.DOTALL)

with open("message.txt", "r", encoding="utf-8") as file:
    log_data = file.read()

# Parse log data again
parsed_data_fixed = [
    {
        "epoch": int(match[0]),
        "accuracy": float(match[1]),
        "f1_score": float(match[2]),
        "loss": float(match[3]),
        "precision": float(match[4]),
        "recall": float(match[5]),
        "val_accuracy": float(match[6]),
        "val_f1_score": float(match[7]),
        "val_loss": float(match[8]),
        "val_precision": float(match[9]),
        "val_recall": float(match[10]),
    }
    for match in pattern_fixed.findall(log_data)
]

# Create DataFrame
df_fixed = pd.DataFrame(parsed_data_fixed)

# Plotting the corrected data
plt.figure(figsize=(12, 6))
plt.plot(df_fixed["epoch"], df_fixed["accuracy"], label="Accuracy")
plt.plot(df_fixed["epoch"], df_fixed["f1_score"], label="F1 Score")
plt.plot(df_fixed["epoch"], df_fixed["val_accuracy"], label="Val Accuracy")
plt.plot(df_fixed["epoch"], df_fixed["val_f1_score"], label="Val F1 Score")

# Enlarged text
plt.xlabel("Epoch", fontsize=24)
plt.ylabel("Score", fontsize=24)
plt.title("Training & Validation Metrics", fontsize=28)
plt.legend(fontsize=18)
plt.grid(True)

# Also increase tick label font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.show()
