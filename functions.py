import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from IPython.display import Markdown


def plot_importance_xgb(model):
    fig, ax = plt.subplots(figsize=(10, 14))
    xgb.plot_importance(
        model,
        ax=ax,
        height=0.85,
        importance_type="gain",  # ”Gain” is the average gain of splits which use the feature
        title="Feature Importance",
        xlabel="Importance",
        values_format="{v:.2f}",
    )
    plt.gca()
    plt.margins(y=0)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def plot_importance_rf(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 14))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance (%)")
    plt.title("Feature Importance")
    plt.grid(True, linestyle="--", linewidth=0.5)
    for i, v in enumerate(importances[indices]):
        plt.text(v + 0.001, i, f"{v:.4f}", va="center", color="black")
    plt.margins(y=0)
    plt.tight_layout()
    plt.show()


def plot_nn(history):
    train_accuracy = history.history["accuracy"]
    train_loss = history.history["loss"]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    c1 = "tab:blue"
    c2 = "tab:orange"
    ax1.plot(
        train_accuracy, label=f"Training Accuracy = {train_accuracy[-1]:.3}", color=c1
    )
    ax2 = ax1.twinx()
    ax2.plot(train_loss, label=f"Training Loss= {train_loss[-1]:.3}", color=c2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")

    ax1.tick_params(axis="y", labelcolor=c1)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.title("Training and Validation Metrics")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()


def metrics_markdown(roc, ks, cvm):
    ks_symbol = "✅" if ks < 0.09 else "❌"
    cvm_symbol = "✅" if cvm < 0.002 else "❌"
    table = f"""
| Metric | Value | Status |
|---|---|---|
| ROC | {roc:.5} |   |
| KS  | {ks:.5} | {ks_symbol} |
| CVM | {cvm:.5} | {cvm_symbol} |
"""
    return Markdown(table)


def plot_roc_curve(y_true, y_scores, model):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, label=f"{model}: ROC curve (area = {str(auc_score)[:5]})" % auc_score
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.95, 1.002])

    return auc_score


def models_metrics_markdown(metrics):
    table = "| Model | ROC | KS | CVM |\n"
    table += "|---|---|---|---| \n"

    for model_name, model_metrics in metrics.items():
        roc = model_metrics["ROC"]
        ks = model_metrics["KS"]
        cvm = model_metrics["CVM"]

        best_roc = max(metrics.values(), key=lambda x: x["ROC"])["ROC"]
        best_ks = min(metrics.values(), key=lambda x: x["KS"])["KS"]
        best_cvm = min(metrics.values(), key=lambda x: x["CVM"])["CVM"]

        roc = f"**{roc:.5f}**" if roc == best_roc else f"{roc:.5f}"
        ks = f"**{ks:.5f}**" if ks == best_ks else f"{ks:.5f}"
        cvm = f"**{cvm:.5f}**" if cvm == best_cvm else f"{cvm:.5f}"

        table += f"| {model_name} | {roc} | {ks} | {cvm} |\n"

    return Markdown(table)


def plot_mass_corr(train, mass, stds=2):
    start = mass.mean() - (mass.std() * stds)
    end = mass.mean() + (mass.std() * stds)
    train["label"] = train["signal"].apply(lambda x: "Sinal" if x == 1 else "Fundo")
    sns.scatterplot(data=train, x="mass", y=mass, s=0.5, hue="label")
    plt.xlim(start, end)
    plt.ylim(start, end)
    plt.ylabel("mass_calc")
    plt.axvline(1777, color="black", linestyle="--", label="Massa do Muon (1777 MeV)")
    plt.axhline(1777, color="black", linestyle="--")
    plt.legend(markerscale=10)
    plt.tight_layout()
    plt.show()
