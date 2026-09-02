import numpy as np
from ucimlrepo import fetch_ucirepo
from setupdata import setup_data
from WFKNN.WFKNN_pipeline import WFKNN

def run_pipeline():
    print("1. THỬ NGHIỆM TRÊN SONAR DATASET (ID: 151)")
    sonar = fetch_ucirepo(id=151) 
    X_tr, X_te, y_tr, y_te = setup_data(sonar, train_ratio=0.7, random_state=42)
    
    m1 = WFKNN(n_neighbors=5, normalize=False, use_mahalanobis_backprop=False).fit(X_tr, y_tr)
    m2 = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=False).fit(X_tr, y_tr)
    m3 = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=True).fit(
        X_tr, y_tr, 
        X_val=X_te,
        y_val=y_te, 
        epochs=1500, 
        lr=0.002
    )

    wine_acc = m3.evaluate(X_te, y_te)
    print(f"Musk Accuracy with WFKNN: {wine_acc * 100:.2f}%")
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

if __name__ == "__main__":
    run_pipeline()