import numpy as np
from collections import Counter
from copy import deepcopy

from .Standardization import MinMaxStandardizer
from .Distance import DistanceCalculator
from .Loss import HingeLossCalculator
from .Plotter import WFKNNPlotter

class WFKNN:
    def __init__(
            self, 
            n_neighbors: int = 5, 
            normalize: bool = True, 
            use_mahalanobis_backprop: bool = True,
            lr: float = 0.01,
            epochs: int = 80
        ):
            self.n_neighbors = n_neighbors
            self.normalize = normalize
            self.use_mahalanobis_backprop = use_mahalanobis_backprop
            self.lr = lr
            self.epochs = epochs
            
            self.standardizer = MinMaxStandardizer() if normalize else None
            self.dist_calc = DistanceCalculator()
            self.loss_calc = HingeLossCalculator()
            self.plotter = WFKNNPlotter(output_dir="Plot") 
            
            self.w = None
            self.sigma2 = None
            self.loss_hist = []
            self.X_train = None
            self.y_train = None
            self.loss_hist = []
            self.val_loss_hist = []
            self.train_acc_hist = []
            self.val_acc_hist = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, 
            epochs: int = None, lr: float = None):
        
        if epochs is not None: self.epochs = epochs
        if lr is not None: self.lr = lr
        
        np.random.seed(42) 
        if self.normalize:
            self.X_train = self.standardizer.fit_transform(X_train)
            X_val_proc = self.standardizer.transform(X_val) if X_val is not None else None
        else:
            self.X_train = X_train.copy()
            X_val_proc = X_val.copy() if X_val is not None else None
            
        self.y_train = y_train.copy()
        d = self.X_train.shape[1]

        if self.use_mahalanobis_backprop:
            self.sigma2 = np.var(self.X_train, axis=0) + 1e-8
            self.w = np.ones(d) / d
            self.loss_hist = []
            self.val_loss_hist = []
            self.train_acc_hist = []
            self.val_acc_hist = []

            best_w = self.w.copy()
            best_val_acc = -1.0

            print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Train Acc':<12} | {'Val Acc':<12}")
            for epoch in range(1, self.epochs + 1):
                total_loss = 0.0
                for i in range(len(self.X_train)):
                    xq, yq = self.X_train[i], self.y_train[i]
                    pos = self.X_train[self.y_train == yq]
                    pos = pos[~np.all(pos == xq, axis=1)]
                    neg = self.X_train[self.y_train != yq]
                    if len(pos) == 0: continue
                    xp = pos[np.random.randint(len(pos))]
                    xn = neg[np.random.randint(len(neg))]
                    dp = self.dist_calc.mahalanobis(xq, xp, self.w, self.sigma2)
                    dn = self.dist_calc.mahalanobis(xq, xn, self.w, self.sigma2)
                    L = self.loss_calc.compute_loss(dp, dn)
                    if L > 0:
                        grad = self.loss_calc.compute_gradient(xq, xp, xn, dp, dn, self.sigma2)
                        self.w -= self.lr * grad
                    total_loss += L
                self.w = np.clip(self.w, 1e-8, None)
                self.w /= self.w.sum()
                self.loss_hist.append(total_loss)
                train_preds = self.predict(X_train)
                t_acc = np.mean(train_preds == y_train) * 100
                self.train_acc_hist.append(t_acc)
                if X_val is not None and y_val is not None:
                    val_preds = self.predict(X_val)
                    v_acc = np.mean(val_preds == y_val) * 100
                    self.val_acc_hist.append(v_acc)

                    if v_acc > best_val_acc:
                        best_val_acc = v_acc
                        best_w = self.w.copy()

                    v_loss = 0.0
                    for i in range(len(X_val_proc)):
                        xq, yq = X_val_proc[i], y_val[i]
                        pos = self.X_train[self.y_train == yq]
                        neg = self.X_train[self.y_train != yq]
                        if len(pos) > 0 and len(neg) > 0:
                            dp = self.dist_calc.mahalanobis(xq, pos[0], self.w, self.sigma2)
                            dn = self.dist_calc.mahalanobis(xq, neg[0], self.w, self.sigma2)
                            v_loss += self.loss_calc.compute_loss(dp, dn)
                    self.val_loss_hist.append(v_loss)
                    v_loss_str = f"{v_loss:.4f}"
                    v_acc_str = f"{v_acc:.2f}%"
                else:
                    v_loss_str = "N/A"
                    v_acc_str = "N/A"
                print(f"{epoch:<8} | {total_loss:<12.4f} | {v_loss_str:<12} | {t_acc:<11.2f}% | {v_acc_str:<12}")

            if X_val is not None and y_val is not None:
                self.w = best_w.copy()

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_te = self.standardizer.transform(X_test) if self.normalize else X_test
        predictions = []

        for xq in X_te:
            if self.use_mahalanobis_backprop:
                dists = [(self.dist_calc.mahalanobis(xq, xi, self.w, self.sigma2), yi) 
                         for xi, yi in zip(self.X_train, self.y_train)]
            else:
                dists = [(self.dist_calc.euclidean(xq, xi), yi) 
                         for xi, yi in zip(self.X_train, self.y_train)]
            
            dists.sort(key=lambda x: x[0])
            pred_label = Counter([y for _, y in dists[:self.n_neighbors]]).most_common(1)[0][0]
            predictions.append(pred_label)

        return np.array(predictions)
    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.array(y_true) == np.array(y_pred))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.predict(X_test)
        return self.accuracy_score(y_test, y_pred) 

    def plot_results(self, X_test: np.ndarray = None, y_test: np.ndarray = None, preds_dict: dict = None, X_raw: np.ndarray = None):
        if self.use_mahalanobis_backprop:
            self.plotter.plot_loss(self.loss_hist)
            self.plotter.plot_feature_weights(self.w)
        if self.X_train is not None and self.y_train is not None:
            self.plotter.plot_pca(self.X_train, self.y_train)
            self.plotter.plot_feature_correlation(self.X_train)
            self.plotter.plot_class_distribution(self.y_train)
            self.plotter.plot_pca_variance(self.X_train)
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            self.plotter.plot_confusion_matrix(y_test, y_pred)

            if preds_dict is not None:
                self.plotter.plot_metrics_comparison(y_test, preds_dict)
                self.plotter.plot_accuracy_comparison(y_test, preds_dict)
        if X_raw is not None and self.X_train is not None:
            self.plotter.plot_normalization_effect(X_raw, self.X_train)
        self.plotter.plot_loss_train_val(self.loss_hist, self.val_loss_hist )
        self.plotter.plot_accuracy_train_val(self.train_acc_hist, self.val_acc_hist)

    def clone(self):
        return deepcopy(self)