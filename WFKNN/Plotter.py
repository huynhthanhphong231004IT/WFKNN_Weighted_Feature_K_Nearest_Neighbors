import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

class WFKNNPlotter:
    def __init__(self, output_dir: str = "Plot"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        mpl.rcParams["font.family"] = "Times New Roman"
        mpl.rcParams["font.size"] = 10

    def _save_fig(self, save_name: str):
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[AutoSave] Đã lưu: {save_path}")

    def plot_loss(self, loss_hist: list, save_name: str = "1_hinge_loss.png"):
        if not loss_hist: return
        plt.figure(figsize=(6, 4))
        plt.plot(loss_hist, color='red', linewidth=1.5)
        plt.title("Hinge Loss qua các Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        self._save_fig(save_name)

    def plot_feature_weights(self, weights: np.ndarray, save_name: str = "2_feature_weights.png"):
        if weights is None: return
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(weights)), weights, color='skyblue', edgecolor='black')
        plt.title("Trọng số các Feature đã học (Mahalanobis Weights)")
        plt.xlabel("Feature Index")
        plt.ylabel("Weight Value")
        plt.grid(alpha=0.3)
        self._save_fig(save_name)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "WFKNN", save_name: str = "3_confusion_matrix.png"):
        labels = np.unique(y_true)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        im = plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix ({model_name})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        self._save_fig(save_name)

    def plot_pca(self, X: np.ndarray, y: np.ndarray, save_name: str = "4_pca_projection.png"):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        labels = np.unique(y)
        plt.figure(figsize=(6, 5))
        for label in labels:
            idx = (y == label)
            plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Class {label}", alpha=0.75)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title("Dataset Distribution (PCA 2D)")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_fig(save_name)

    def plot_metrics_comparison(self, y_true: np.ndarray, preds_dict: dict, save_name: str = "5_metrics_matrix.png"):
        models = list(preds_dict.keys())
        metrics = ["Precision", "Recall", "F1-score"]
        matrix_data = []

        for y_pred in preds_dict.values():
            p = precision_score(y_true, y_pred, average="macro", zero_division=0)
            r = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            matrix_data.append([p, r, f1])

        M = np.array(matrix_data)
        plt.figure(figsize=(7, 4.5))
        im = plt.imshow(M, cmap="YlGnBu", vmin=0, vmax=1)
        plt.xticks(range(len(metrics)), metrics)
        plt.yticks(range(len(models)), models)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="black")
        plt.title("Precision–Recall–F1 Matrix Comparison")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        self._save_fig(save_name)

    def plot_feature_correlation(self, X: np.ndarray, save_name: str = "6_feature_correlation.png"):
        plt.figure(figsize=(8, 6))
        corr = np.corrcoef(X, rowvar=False)
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        self._save_fig(save_name)

    def plot_class_distribution(self, y: np.ndarray, save_name: str = "7_class_distribution.png"):
        labels, counts = np.unique(y, return_counts=True)
        plt.figure(figsize=(6, 4))
        plt.bar([str(l) for l in labels], counts, color='teal', alpha=0.8)
        plt.title("Class Sample Distribution")
        plt.xlabel("Class Label")
        plt.ylabel("Number of Samples")
        plt.grid(alpha=0.3, axis='y')
        self._save_fig(save_name)

    def plot_pca_variance(self, X: np.ndarray, save_name: str = "8_pca_explained_variance.png"):
        pca = PCA().fit(X)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--', color='purple')
        plt.axhline(y=0.95, color='r', linestyle=':', label='95% Cutoff Threshold')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance Ratio")
        plt.legend()
        plt.grid(alpha=0.3)
        self._save_fig(save_name)

    def plot_accuracy_comparison(self, y_true: np.ndarray, preds_dict: dict, save_name: str = "9_accuracy_comparison.png"):
        models = list(preds_dict.keys())
        accs = [np.mean(y_true == y_pred) * 100 for y_pred in preds_dict.values()]
        
        plt.figure(figsize=(7, 4))
        bars = plt.bar(models, accs, color=['#4C72B0', '#55A868', '#C44E52'])
        plt.ylabel("Accuracy (%)")
        plt.title("Model Accuracy Comparison")
        plt.ylim(0, 105)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom')
        plt.grid(alpha=0.3, axis='y')
        self._save_fig(save_name)

    def plot_normalization_effect(self, X_raw: np.ndarray, X_scaled: np.ndarray, feature_idx: int = 0, save_name: str = "10_normalization_effect.png"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        
        axes[0].boxplot(X_raw[:, feature_idx])
        axes[0].set_title(f"Original Feature {feature_idx}")
        axes[0].set_ylabel("Raw Values")
        axes[0].grid(alpha=0.3)

        axes[1].boxplot(X_scaled[:, feature_idx])
        axes[1].set_title(f"Scaled Feature {feature_idx}")
        axes[1].set_ylabel("Min-Max Values [0, 1]")
        axes[1].grid(alpha=0.3)

        self._save_fig(save_name)