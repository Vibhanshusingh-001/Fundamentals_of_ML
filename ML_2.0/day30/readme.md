# FunctionTransformer

It allows you to apply a custom function to data.

log1p -- to handle log zero if any values is zero

```python
trf = FunctionTransformer(func=np.log1p)# log1p -- to handle log zero if any values is zero

trf2 = ColumnTransformer([('log',FunctionTransformer(np.log1p),['Fare'])],remainder='passthrough')

X_train_transformed2 = trf2.fit_transform(X_train)
X_test_transformed2 = trf2.transform(X_test)
