# Task 6: K-Nearest Neighbors (KNN) – Iris Flower Classification

## Objective
To classify Iris flowers into different species using the K-Nearest Neighbors (KNN) algorithm. This task involves dataset preprocessing, model training, hyperparameter tuning, and evaluation.

---

## Tools Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Dataset
Dataset used: [Iris Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/iris)  

- **Target Variable:**  
  - `Iris-setosa`  
  - `Iris-versicolor`  
  - `Iris-virginica`  
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width (all in cm)

---

## Steps Performed

### 1. Data Loading and Exploration
- Loaded dataset from `Iris.csv`.
- Dropped `Id` column.
- Checked for missing values and basic statistics.

### 2. Data Preprocessing
- Selected features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) and target (`Species`).
- Standardized features using `StandardScaler`.

### 3. Train-Test Split
- Split dataset into 80% training and 20% testing sets.

### 4. Model Training with KNN
- Trained `KNeighborsClassifier` with different values of `K` (1 to 10).
- Plotted accuracy scores for each `K` to select the best value.

### 5. Final Model with Best K
- Selected best `K` based on highest accuracy.
- Trained and tested the final KNN model.

### 6. Model Evaluation
- Calculated accuracy score.
- Generated confusion matrix.
- Created classification report with precision, recall, and F1-score.

### 7. Visualization
- Plotted accuracy vs. K value.
- (Optional) Plotted decision boundaries using only sepal length and sepal width for visualization purposes.

---

---

## Key Insights
1. KNN performed very well on the Iris dataset, achieving high accuracy.
2. Standardization of features improved model performance.
3. Choosing the right `K` is important — too small can lead to overfitting, too large can underfit.
4. The model is simple, interpretable, and effective for small datasets.

---

## Prepared By
Tanuja Deshmukh  
AI & ML Internship – Task 6
