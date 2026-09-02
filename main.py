import numpy as np
from ucimlrepo import fetch_ucirepo
from setupdata import setup_data
from WFKNN.WFKNN_pipeline import WFKNN

def run_pipeline():
    print("1. THỬ NGHIỆM TRÊN WINE DATASET (ID: 109)")
    wine = fetch_ucirepo(id=109) 
    X_tr, X_te, y_tr, y_te = setup_data(wine, train_ratio=0.7, random_state=42)
    m1 = WFKNN(n_neighbors=5, normalize=False, use_mahalanobis_backprop=False).fit(X_tr, y_tr)
    m2 = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=False).fit(X_tr, y_tr)
    m3 = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=True).fit(X_tr, y_tr)

    wine_acc = m3.evaluate(X_te, y_te)
    print(f"Wine Accuracy with WFKNN: {wine_acc * 100:.2f}%")
    preds_dict_wine = {
        "KNN (No Norm)": m1.predict(X_te),
        "KNN (Min-Max)": m2.predict(X_te),
        "WFKNN (Mahalanobis)": m3.predict(X_te)
    }
    print("\n--- Wine Accuracy Results ---")
    print(f"- KNN (No Norm)        : {m1.evaluate(X_te, y_te) * 100:.2f}%")
    print(f"- KNN (Min-Max)        : {m2.evaluate(X_te, y_te) * 100:.2f}%")
    print(f"- WFKNN (Mahalanobis)  : {wine_acc * 100:.2f}%")

    m3.plot_results(X_test=X_te, y_test=y_te, preds_dict=preds_dict_wine, X_raw=X_tr)

    print("2. THỬ NGHIỆM TRÊN BREAST CANCER DATASET (ID: 17)")
    cancer = fetch_ucirepo(id=17)
    X_tr_bc, X_te_bc, y_tr_bc, y_te_bc = setup_data(cancer, train_ratio=0.8, random_state=42)

    m1_bc = WFKNN(n_neighbors=5, normalize=False, use_mahalanobis_backprop=False).fit(X_tr_bc, y_tr_bc)
    m2_bc = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=False).fit(X_tr_bc, y_tr_bc)
    m3_bc = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=True).fit(X_tr_bc, y_tr_bc)

    preds_dict_bc = {
        "KNN (No Norm)": m1_bc.predict(X_te_bc),
        "KNN (Min-Max)": m2_bc.predict(X_te_bc),
        "WFKNN (Mahalanobis)": m3_bc.predict(X_te_bc)
    }

    print("\n--- Breast Cancer Accuracy Results ---")
    print(f"- KNN (No Norm)        : {m1_bc.evaluate(X_te_bc, y_te_bc) * 100:.2f}%")
    print(f"- KNN (Min-Max)        : {m2_bc.evaluate(X_te_bc, y_te_bc) * 100:.2f}%")
    print(f"- WFKNN (Mahalanobis)  : {m3_bc.evaluate(X_te_bc, y_te_bc) * 100:.2f}%")

    m3_bc.plot_results(X_test=X_te_bc, y_test=y_te_bc, preds_dict=preds_dict_bc, X_raw=X_tr_bc)

if __name__ == "__main__":
    run_pipeline()