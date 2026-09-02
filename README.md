<h2 align="center">
  Author: Huynh Thanh Phong (ReoRioll)
</h2>

<p align="center">
   Computer Science of College of Information and Communication Technology of Can Tho University (Course 48)<br>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Researchs:</b> Artificial Intelligence in Education - Mathematics in Deep Learning and Machine Learning<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<mark><b><b>Name Project:</b></b> </mark> WFKNN: Weighted-Feature K-Nearest Neighbors<br>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Timeline:</b> 03/2026 at Computer science department
</p>
<p align="center">
   <b>Presional link Information</b>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Facbook: https://www.facebook.com/huynh.thanh.phong.561667 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Kaggle: https://www.kaggle.com/reorioll <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Youtobe: https://www.youtube.com/@ReoRioll-2304CICTCTU <br>
</p>
<br>

<h3 align="left">
  <span style="color:#8B4513;">
    <b>Guide to training WFKNN with the Sonar dataset (ID = 151 on fetch_ucirepo)</b>
  </span>
</h3>

<p>
  Step 1. Clone module from git
</p>

```python
!git clone https://github.com/huynhthanhphong231004IT/WFKNN_Weighted_Feature_K_Nearest_Neighbors.git
```

<p>
  Step 2. Install all Python libraries listed in the requirements.txt file.
</p>

```python
!pip install -r requirements.txt
```

<p>
  Step 3. Download the data and start dividing the train/val set (setupdata.py)
</p>

```python
import numpy as np
def setup_data(
    dataset, 
    train_ratio: float = 0.7, 
    random_state: int = 42,
    return_numpy: bool = True
):
    X = dataset.data.features
    y = dataset.data.targets

    if return_numpy:
        X = X.values.astype(float)
        y = y.values.ravel()

    np.random.seed(random_state)
    n_samples = len(X)
    idx = np.random.permutation(n_samples)
    n_train = int(train_ratio * n_samples)

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    if return_numpy:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test
```


```python
from ucimlrepo import fetch_ucirepo
from setupdata import setup_data

sonar = fetch_ucirepo(id=151) 
X_tr, X_te, y_tr, y_te = setup_data(sonar, train_ratio=0.7, random_state=42)
    
```

<p>
  Step 4. Training and comparing the results of three models.
</p>

```python
m1 = WFKNN(n_neighbors=5, normalize=False, use_mahalanobis_backprop=False).fit(X_tr, y_tr)
m2 = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=False).fit(X_tr, y_tr)
m3 = WFKNN(n_neighbors=5, normalize=True, use_mahalanobis_backprop=True).fit(
  X_tr, y_tr, 
  X_val=X_te,
  y_val=y_te, 
  epochs=1500, 
  lr=0.002)
```

<p>
  Step 4. Evaluate results for the three models.
</p>

```python
print(f"- KNN (No Norm)        : {m1.evaluate(X_te, y_te) * 100:.2f}%")
print(f"- KNN (Min-Max)        : {m2.evaluate(X_te, y_te) * 100:.2f}%")
print(f"- WFKNN (Mahalanobis)  : {m3.evaluate(X_te, y_te) * 100:.2f}%")
```

<p>
  Step 4. Plot the relevant evaluation charts..
</p>

```python
m3.plot_results(X_test=X_te, y_test=y_te, preds_dict=preds_dict_wine, X_raw=X_tr)
```
