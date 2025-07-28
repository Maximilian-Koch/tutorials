## Dimensionality Reduction Tutorial: PCA on Hyperspectral Data

In this tutorial, we build use **Principal Component Analysis (PCA)** for dimensionality reduction on hyperspectral imagery, followed by classification using a **Random Forest**.
---

### Introduction

Hyperspectral images contain hundreds of spectral bands per pixel, resulting in high-dimensional data. Processing and visualizing such data can be challenging due to the **curse of dimensionality**. PCA helps by transforming the data into a lower-dimensional space that retains most of the variance.

**Applications**:

* Preprocessing for classification and clustering
* Visualization of high-dimensional data
* Noise reduction and data compression

---

### 1. Load and Prepare Data

We load the Salinas Valley hyperspectral dataset (MATLAB files) and reshape it into a 2D table where each row is a pixel and each column a spectral band.

```python
import numpy as np
import pandas as pd
import scipy.io

# Read MATLAB file and flatten the image
img = scipy.io.loadmat('dimensionality_dataset_X.mat')['salinas_corrected']  # (rows, cols, bands)
num_bands = img.shape[-1]
flat_img = img.reshape(-1, num_bands)  # flatten to (pixels, bands)
X = pd.DataFrame(flat_img, columns=range(num_bands))  # DataFrame of band values

# Load ground truth labels
y = scipy.io.loadmat('dimensionality_groundtruth.mat')['salinas_gt'].flatten()

print(X.head())  # inspect first rows
```

*Comments*: We structure the data for modeling, preserving all spectral bands.

---

### 2. PCA for Variance Retention

We apply PCA to retain 99% of the variance, reducing dimensionality significantly.

```python
from sklearn.decomposition import PCA

# Apply PCA to retain 99% of variance
pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X)

print('Original dimensions:', X.shape[1])
print('Reduced dimensions:', pca.n_components_)
```

*Explanation*: PCA projects data onto orthogonal axes (principal components) sorted by explained variance. Here, only the top \~3 components are needed to capture 99% variance.

---

### 3. Classification with Random Forest

We compare performance of a Random Forest classifier on raw vs. PCA-transformed data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_pca_train, X_pca_test, _, _ = train_test_split(X_pca, y, test_size=0.2, stratify=y)

# Train on original data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Original Data Classification')
print(classification_report(y_test, y_pred))

# Train on PCA-reduced data
rf_pca = RandomForestClassifier(n_estimators=100)
rf_pca.fit(X_pca_train, y_train)
y_pca_pred = rf_pca.predict(X_pca_test)
print('PCA-Reduced Data Classification')
print(classification_report(y_test, y_pca_pred))
```

*Insights*: Dimensionality reduction often speeds up training with minimal loss in accuracy, especially for high-dimensional inputs.

---

### 4. Visualizing Clusters in 2D

To interpret class separability, we visualize the first two principal components. Due to class imbalance, we undersample minority classes.

```python
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Combine PCA components and labels
df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
df['label'] = y

# Undersample majority classes for balance
balanced = df.groupby('label').apply(
    lambda d: resample(d, replace=False, n_samples=100, random_state=0)
).reset_index(drop=True)

# Scatter plot of PC1 vs PC2
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    balanced['PC1'], balanced['PC2'],
    c=balanced['label'], s=10, edgecolor='k', linewidth=0.2
)
plt.colorbar(scatter, label='Class')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D PCA of Hyperspectral Data (Balanced)')
plt.show()
```

*Interpretation*: Clusters show how different land cover types separate in the reduced space.

---

### Conclusion and Next Steps

* **PCA** effectively reduces dimensions, retaining most variance.
* **Random Forest** on PCA data can match original accuracy with faster computation.
* **Visualization** aids interpretability of complex data.

**Further Applications**:

* Use other techniques like t-SNE or UMAP for non-linear structure.
* Automate variance threshold selection.
* Integrate into end-to-end remote sensing pipelines.