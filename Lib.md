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

### Summary of Main Libraries Used

| Library                          | Main Use                                                              |
| -------------------------------- | --------------------------------------------------------------------- |
| Scikit-learn (`sklearn`)         | Machine learning model building, training, prediction, and evaluation |
| Matplotlib (`matplotlib.pyplot`) | Visualization and plotting graphs                                     |
| `pickle`                         | Saving and loading trained ML models                                  |
