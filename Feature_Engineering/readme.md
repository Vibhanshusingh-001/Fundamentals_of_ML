# Feature Engineering

# Do train test split before standard scaling.
### Mostly we prefer standardisation.
### In standardisation mean should be Zero and standard deviation should be 1.
### We do normalisation when we know that min and max value of our data 


### We have to transform train and test both after fitting standard scaler on out train data.

```python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

```

<img width="1500" height="806" alt="Screenshot 2026-05-12 062804" src="https://github.com/user-attachments/assets/37c80ae9-a1f8-4672-b7cf-76c8e7404922" />



---
Feature Engineering is the process of transforming raw data into meaningful input features that improve the performance of Machine Learning models.

It is one of the most important steps in Machine Learning because the quality of features directly affects model accuracy.

The diagram divides Feature Engineering into four major parts:

1. Feature Transformation
2. Feature Construction
3. Feature Selection
4. Feature Extraction

---

# 1. Feature Transformation

Feature Transformation means modifying existing features into a better format so that machine learning algorithms can understand them efficiently.

The image shows four important tasks inside Feature Transformation:

* Missing Value Imputation
* Handling Categorical Features
* Outlier Detection
* Feature Scaling



# A. Missing Value Imputation

Real-world datasets often contain missing values.

Example:

| Age | Salary |
| --- | ------ |
| 25  | 50000  |
| 30  | NaN    |
| 28  | 60000  |

Here `NaN` means missing value.

Machine learning models usually cannot handle missing values directly.

So we fill them using imputation methods.



## Common Techniques

### 1. Mean Imputation

Replace missing values with the average.

Mean Formula:


$$
\bar{x} = \frac{x_1+x_2+\cdots+x_n}{n}
$$


Example:

Salary values = 50000 and 60000

$$
\frac{50000 + 60000}{2} = 55000
$$


Missing value becomes 55000.



### 2. Median Imputation

Use the middle value.

Useful when data contains outliers.



### 3. Mode Imputation

Used for categorical data.

Example:

| Color |
| ----- |
| Red   |
| Blue  |
| Red   |
| NaN   |

Most frequent value = Red

Missing value becomes Red.



### 4. Forward Fill / Backward Fill

Mostly used in time-series data.



# B. Handling Categorical Features

Machine Learning models work with numbers, not text categories.

Example:

| City    |
| ------- |
| Delhi   |
| Mumbai  |
| Chennai |

We convert categories into numerical form.



## 1. Label Encoding

Each category gets a number.

| City    | Encoded |
| ------- | ------- |
| Delhi   | 0       |
| Mumbai  | 1       |
| Chennai | 2       |

Problem:
The model may think Chennai > Mumbai > Delhi because of numbering.



## 2. One-Hot Encoding

Creates separate binary columns.

| City_Delhi | City_Mumbai | City_Chennai |
| ---------- | ----------- | ------------ |
| 1          | 0           | 0            |
| 0          | 1           | 0            |
| 0          | 0           | 1            |

This is widely used.



## 3. Ordinal Encoding

Used when categories have order.

Example:

| Education    |
| ------------ |
| High School  |
| Graduate     |
| Postgraduate |

Encoding:

* High School = 1
* Graduate = 2
* Postgraduate = 3



# C. Outlier Detection

Outliers are abnormal or extreme values.

Example:

| Salary  |
| ------- |
| 30000   |
| 35000   |
| 40000   |
| 1000000 |

Here 1000000 is an outlier.

Outliers can negatively affect machine learning models.



## Methods to Detect Outliers

### 1. Z-Score Method

Measures how far a value is from the mean.

Formula:


$$
z = \frac{x-\mu}{\sigma}
$$


Where:

* (x) = data point
* (\mu) = mean
* (\sigma) = standard deviation

If:


$$
|z| > 3
$$


then the value is usually considered an outlier.



### 2. IQR Method

Interquartile Range:


$$
IQR = Q3 - Q1
$$


Outlier conditions:


$$
x < Q1 - 1.5(IQR)
$$


or


$$
x > Q3 + 1.5(IQR)
$$




# D. Feature Scaling

Different features may have different ranges.

Example:

| Age | Salary |
| --- | ------ |
| 25  | 50000  |
| 30  | 80000  |

Salary values are much larger than Age values.

Some algorithms like:

* KNN
* SVM
* Logistic Regression
* Neural Networks

are sensitive to scale.



## 1. Min-Max Scaling

Transforms values between 0 and 1.

Formula:


$$
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
$$




## 2. Standardization

Transforms data with mean 0 and standard deviation 1.

Formula:


$$
x' = \frac{x-\mu}{\sigma}
$$




# 2. Feature Construction

Feature Construction means creating new features from existing features.

This helps the model learn hidden relationships.



## Example 1: Date Features

Original column:

| Date       |
| ---------- |
| 2025-05-10 |

Constructed features:

* Day
* Month
* Year
* Weekday
* Weekend



## Example 2: BMI Calculation

From:

* Height
* Weight

Construct:

BMI

Formula:


$$
BMI = \frac{weight}{height^2}
$$




## Example 3: Interaction Features

Suppose:

* Length
* Width

Construct:

Area

Formula:


$$
Area = Length \times Width
$$



# 3. Feature Selection

Feature Selection means choosing only the most useful features and removing irrelevant ones.

Not all columns help the model.

Some may:

* Add noise
* Increase training time
* Cause overfitting



# Why Feature Selection is Important

Benefits:

* Faster training
* Better accuracy
* Reduced overfitting
* Simpler models



# Types of Feature Selection

## 1. Filter Methods

Uses statistical techniques.

Examples:

* Correlation
* Chi-Square Test
* ANOVA

Example:

If two features are highly correlated, remove one.



## 2. Wrapper Methods

Tests multiple feature combinations.

Examples:

* Forward Selection
* Backward Elimination
* Recursive Feature Elimination (RFE)

These are accurate but computationally expensive.



## 3. Embedded Methods

Feature selection happens during model training.

Examples:

* Lasso Regression
* Decision Trees
* Random Forest

Lasso uses regularization.

Formula:


$$
Loss = RSS + \lambda \sum_{j=1}^{p} |\beta_j|
$$


It can shrink some coefficients to zero, effectively removing features.



# 4. Feature Extraction

Feature Extraction creates entirely new reduced representations of data.

Instead of selecting existing features, it generates new compressed features.

Used especially for:

* High-dimensional data
* Images
* Text
* Genomics
* Bioinformatics



# Example

Suppose a dataset has 1000 features.

Feature extraction may reduce them to 20 important components.



# Common Feature Extraction Techniques

## 1. PCA (Principal Component Analysis)

Reduces dimensions while preserving maximum variance.

Main idea:

Transform data into new orthogonal axes called Principal Components.



## 2. t-SNE

Used for visualization of high-dimensional data.



## 3. Word Embeddings (NLP)

Converts text into vectors.

Examples:

* Word2Vec
* GloVe
* BERT embeddings


## 4. CNN Feature Extraction

In Deep Learning, Convolutional Neural Networks automatically extract image features such as:

* edges
* textures
* shapes



# Difference Between Feature Selection and Feature Extraction

| Feature Selection         | Feature Extraction            |
| ------------------------- | ----------------------------- |
| Selects existing features | Creates new features          |
| Original features remain  | Original features transformed |
| Easier to interpret       | Harder to interpret           |
| Faster                    | More computational            |

Example:

* Feature Selection → Removing irrelevant columns
* Feature Extraction → PCA



## Complete Flow of Feature Engineering
```
Raw Data
↓
Handle Missing Values
↓
Encode Categorical Features
↓
Detect Outliers
↓
Scale Features
↓
Construct New Features
↓
Select Important Features
↓
Extract Better Representations
↓
Train Machine Learning Model

```

# Real-World Example

Suppose you are predicting house prices.

Raw features:

* Area
* Bedrooms
* City
* Construction Year

Feature engineering steps:

1. Fill missing area values
2. Encode city names
3. Remove abnormal house prices
4. Scale numerical columns
5. Create house age feature:


$$
House\ Age = Current\ Year - Construction\ Year
$$


6. Select important features
7. Train model



# Key Idea

Good feature engineering can improve model performance more than changing algorithms.

A simple algorithm with good features often performs better than a complex algorithm with poor features.
