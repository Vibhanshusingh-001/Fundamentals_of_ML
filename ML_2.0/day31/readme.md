## What is `PowerTransformer()`?

`PowerTransformer()` is a preprocessing technique in **Scikit-learn** used to make numerical data **more normally distributed (Gaussian)**.

Real-world data is often **skewed**:

* **Right skewed** → long tail on right (common)
* **Left skewed** → long tail on left

Many ML algorithms work better when data is closer to a **normal distribution**.

Examples:

* Linear Regression
* Logistic Regression
* KNN
* SVM
* PCA

### Import

```python
from sklearn.preprocessing import PowerTransformer
```

### Syntax

```python
pt = PowerTransformer(method='yeo-johnson')
```

Available methods:

| Method        | Negative values allowed? |
| ------------- | -----------------------: |
| `yeo-johnson` |                    ✅ Yes |
| `box-cox`     |     ❌ No (positive only) |

---

## Why do we need Power Transformation?

Suppose salary data is highly skewed:

```python
salary = [1000, 1200, 1500, 1800, 2000, 100000]
```

The value `100000` dominates the distribution.

This can hurt model performance.

Power transformation compresses large values and makes the distribution more balanced.

### Before transformation

```text
1000 1200 1500 1800 2000 ..........100000
```

Highly skewed

### After transformation

```text
1000 1200 1500 1800 2000 3000
```

Less skewed (conceptually)



## What is Box-Cox Transformation?

Box-Cox transformation is a mathematical technique used to make skewed data more normally distributed.

The Box-Cox formula is:

$$
y(\lambda)=
\begin{cases}
\frac{x^\lambda - 1}{\lambda}, & \lambda \neq 0 \\
\log(x), & \lambda = 0
\end{cases}
$$

### Where:
- $x$ = original feature value  
- $\lambda$ (lambda) = transformation parameter chosen automatically  
- $y(\lambda)$ = transformed value  

### Interpretation of Lambda ($\lambda$)

| Lambda Value | Transformation |
|---------------|----------------|
| $\lambda = 1$ | No transformation |
| $\lambda = 0$ | Log transformation |
| $\lambda = 0.5$ | Square root transformation |
| $\lambda = 2$ | Square transformation |


because it cannot handle **zero or negative numbers**.
**Box-Cox** is a mathematical power transformation used to convert skewed data into something closer to a normal distribution.

### Important Rule

Box-Cox works **only when values are positive (> 0)**.

This will work:

```python
[1, 2, 3, 10, 50]
```

This will fail:

```python
[-1, 0, 2, 10]
```

because Box-Cox cannot handle:

* `0`
* negative numbers

### Example

```python
from sklearn.preprocessing import PowerTransformer
import pandas as pd

df = pd.DataFrame({
    'income': [1000, 1200, 1500, 2000, 10000]
})

pt = PowerTransformer(method='box-cox')

df['income_transformed'] = pt.fit_transform(df[['income']])

print(df)
```

### What happens internally?

The algorithm automatically finds the **best λ (lambda)** value.

Different λ values produce different transformations:

| λ (lambda) | Transformation |
| ---------- | -------------- |
| λ = 1      | No change      |
| λ = 0      | Log transform  |
| λ = 0.5    | Square root    |
| λ = 2      | Square         |

The model chooses the λ that best reduces skewness.

---

## What is Yeo-Johnson Transformation?

`Yeo-Johnson` is an improved version of Box-Cox.

It works with:

✅ positive numbers
✅ zero values
✅ negative numbers

Example:

```python
[-10, -5, 0, 5, 20]
```

This is why it is the **default method** in `PowerTransformer()`.

```python
pt = PowerTransformer(method='yeo-johnson')
```

---

## Difference Between Box-Cox and Yeo-Johnson

| Feature                         | Box-Cox | Yeo-Johnson |
| ------------------------------- | ------: | ----------: |
| Positive values                 |       ✅ |           ✅ |
| Zero values                     |       ❌ |           ✅ |
| Negative values                 |       ❌ |           ✅ |
| More flexible                   |       ❌ |           ✅ |
| Default in `PowerTransformer()` |       ❌ |           ✅ |

---

## Example Comparison

### Box-Cox

```python
pt = PowerTransformer(method='box-cox')
X_transformed = pt.fit_transform(X)
```

Only if all values are positive.

### Yeo-Johnson

```python
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)
```

Safer because it handles negative values.

---

## Example with `ColumnTransformer`

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

trf = ColumnTransformer([
    ('power', PowerTransformer(method='box-cox'), ['Fare'])
], remainder='passthrough')

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
```

Here:

* `Fare` column gets power transformation
* remaining columns stay unchanged

---

### When should you use which?

| Situation                | Best Choice          |
| ------------------------ | -------------------- |
| Only positive values     | `box-cox`            |
| Negative or zero present | `yeo-johnson`        |
| Heavy skewness           | `PowerTransformer()` |
| Mild skewness            | `log1p()`            |

### Simple intuition

Think of Box-Cox as:

> “A smart log transformation that automatically chooses the best mathematical power.”

And `PowerTransformer()` is:

> “An automatic skewness remover.”
