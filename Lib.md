```python

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
```
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
```
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
pickle.dump(model,open('model.pkl','wb'))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
```

```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
```

```python
# ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
```

| Library / Module                       | Function / Class Used      | Purpose                   | What It Does                                                                               |
| -------------------------------------- | -------------------------- | ------------------------- | ------------------------------------------------------------------------------------------ |
| Scikit-learn `sklearn.model_selection` | `train_test_split()`       | Data splitting            | Splits dataset into training and testing sets for model training and evaluation            |
| Scikit-learn `sklearn.linear_model`    | `LogisticRegression()`     | Model creation            | Creates a Logistic Regression classification model                                         |
| Scikit-learn `sklearn.linear_model`    | `fit()`                    | Model training            | Trains the Logistic Regression model using training data (`X_train`, `y_train`)            |
| Scikit-learn `sklearn.linear_model`    | `predict()`                | Prediction                | Predicts output labels/classes for unseen test data (`X_test`)                             |
| Scikit-learn `sklearn.metrics`         | `accuracy_score()`         | Performance evaluation    | Calculates model accuracy (correct predictions / total predictions)                        |
| `pickle` (Python built-in library)     | `dump()`                   | Model saving              | Saves the trained model into a file (`model.pkl`) so it can be reused later                |
| `pickle` (Python built-in library)     | `load()`                   | Model loading             | Loads the saved model without retraining                                                   |
| Scikit-learn `sklearn.metrics`         | `confusion_matrix()`       | Classification evaluation | Creates a confusion matrix showing correct and incorrect predictions                       |
| Scikit-learn `sklearn.metrics`         | `ConfusionMatrixDisplay()` | Visualization             | Creates a visual display object for confusion matrix plotting                              |
| Matplotlib `matplotlib.pyplot` (`plt`) | `plot()`                   | Plotting graphs           | Draws graphs such as ROC curve                                                             |
| Matplotlib `matplotlib.pyplot` (`plt`) | `figure()`                 | Figure creation           | Creates a plotting canvas with specified size                                              |
| Matplotlib `matplotlib.pyplot` (`plt`) | `title()`                  | Add title                 | Adds title to graph                                                                        |
| Matplotlib `matplotlib.pyplot` (`plt`) | `xlabel()`                 | X-axis label              | Labels x-axis                                                                              |
| Matplotlib `matplotlib.pyplot` (`plt`) | `ylabel()`                 | Y-axis label              | Labels y-axis                                                                              |
| Matplotlib `matplotlib.pyplot` (`plt`) | `legend()`                 | Legend display            | Displays graph labels/legend                                                               |
| Matplotlib `matplotlib.pyplot` (`plt`) | `grid()`                   | Grid display              | Adds grid lines to graph                                                                   |
| Matplotlib `matplotlib.pyplot` (`plt`) | `show()`                   | Output visualization      | Displays the final plot                                                                    |
| Scikit-learn `sklearn.metrics`         | `roc_curve()`              | ROC calculation           | Computes False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for ROC curve |
| Scikit-learn `sklearn.metrics`         | `roc_auc_score()`          | Model evaluation          | Calculates AUC (Area Under Curve) score to measure model performance                       |
| Scikit-learn `sklearn.linear_model`    | `predict_proba()`          | Probability prediction    | Returns probability scores for each class instead of only class labels                     |


# LIBrary




| Import Statement                                          | Library / Function       | Use / Purpose                                                                   | Example Use in ML                                            |
| --------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `from sklearn.model_selection import train_test_split`    | `train_test_split()`     | Splits dataset into **training** and **testing** sets                           | Train model on 80% data and test on 20%                      |
| `from sklearn.compose import ColumnTransformer`           | `ColumnTransformer`      | Applies **different preprocessing** to different columns                        | One-hot encode categorical columns and scale numeric columns |
| `from sklearn.impute import SimpleImputer`                | `SimpleImputer`          | Handles **missing values (NaN)**                                                | Replace missing age values with mean                         |
| `from sklearn.preprocessing import OneHotEncoder`         | `OneHotEncoder`          | Converts **categorical data into numerical format** using binary columns        | Convert `Male/Female` → `[1,0]`, `[0,1]`                     |
| `from sklearn.preprocessing import MinMaxScaler`          | `MinMaxScaler`           | Scales numerical data between **0 and 1**                                       | Normalize salary or age values                               |
| `from sklearn.pipeline import Pipeline, make_pipeline`    | `Pipeline`               | Creates a **step-by-step ML workflow**                                          | Imputation → Encoding → Scaling → Model                      |
| `from sklearn.pipeline import Pipeline, make_pipeline`    | `make_pipeline()`        | Shortcut to create a pipeline without naming steps                              | Faster way to build preprocessing + model                    |
| `from sklearn.feature_selection import SelectKBest, chi2` | `SelectKBest`            | Selects **top K important features**                                            | Choose best 10 columns from dataset                          |
| `from sklearn.feature_selection import SelectKBest, chi2` | `chi2`                   | Statistical test used for **feature selection** (categorical/non-negative data) | Find features most related to target variable                |
| `from sklearn.tree import DecisionTreeClassifier`         | `DecisionTreeClassifier` | A **classification algorithm** based on decision rules                          | Predict disease Yes/No                                       |

### Workflow of these components in ML

```text
Raw Data
   ↓
SimpleImputer (fill missing values)
   ↓
OneHotEncoder (convert categorical → numeric)
   ↓
MinMaxScaler (scale numeric features)
   ↓
SelectKBest + chi2 (select best features)
   ↓
DecisionTreeClassifier (train model)
   ↓
Prediction
```

### Example of how they work together

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier

# preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), ['Age', 'Salary']),
    ('cat', OneHotEncoder(), ['Gender', 'City'])
])

# full pipeline
pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('feature_selection', SelectKBest(chi2, k=5)),
    ('model', DecisionTreeClassifier())
])
```

In this example:

* **`ColumnTransformer`** → handles numerical and categorical columns separately
* **`OneHotEncoder`** → converts text categories into numbers
* **`MinMaxScaler`** → normalizes values
* **`SelectKBest(chi2)`** → selects important features
* **`DecisionTreeClassifier`** → builds prediction model
* **`Pipeline`** → combines everything into one workflow
* **`train_test_split`** → splits data for training/testing

