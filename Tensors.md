# Arrays, Vectors, and Tensors 

#### Tensors---> n diamensional array of same thing
#### Diamension = axis = rank
#### 1 D array == 1 D Tensor
#### 1 D tensor are collection of 0 D tensor
#### 2 D tensor are collection of 1 D tensor
#### 1 D tensor --> [2,3,4,5,6] but 5D vector
#### Rank == Axes == Diamensions
#### Shape ---> how many items in perticular axes (2,3)
#### Size ---> How many items =(2,3)=6

# 1. Array

An **array** is a collection of elements stored in an organized way.

It can contain numbers, characters, or other data types.

Arrays are used to store multiple values in a single variable.

## Example of a 1D Array

```python
arr = [1, 2, 3, 4, 5]
```

This is a simple list of numbers.

Representation:

```text
[1 2 3 4 5]
```

---

## Example of a 2D Array

```python
arr_2d = [
    [1, 2, 3],
    [4, 5, 6]
]
```

Representation:

```text
[
 [1 2 3]
 [4 5 6]
]
```

This looks like a table with rows and columns.

---

## Dimensions in Arrays

| Array Type      | Dimensions |
| --------------- | ---------- |
| `[1,2,3]`       | 1D         |
| `[[1,2],[3,4]]` | 2D         |
| `[[[1]]]`       | 3D         |

---

# 2. Vector

A **vector** is a special type of 1D array used in mathematics and machine learning.

A vector has:

* Magnitude (length)
* Direction

In machine learning, vectors are commonly used to represent:

* Features
* Data points
* Embeddings
* Coordinates

---

## Example of a Vector

```python
v = [2, 4, 6]
```

Representation:

```text
v = [2 4 6]
```

This vector has 3 elements.

---

## Real-Life Example

Suppose a student has:

* Math marks = 90
* Science marks = 85
* English marks = 88

This can be represented as a vector:

```python
student = [90, 85, 88]
```

---

## Types of Vectors

### Row Vector

```text
[1 2 3]
```

### Column Vector

```text
[
 1
 2
 3
]
```

---

# 3. Tensor

A **tensor** is a generalized form of arrays and vectors.

A tensor can have:

* 0 dimensions
* 1 dimension
* 2 dimensions
* 3 dimensions
* n dimensions

Tensors are the basic data structure used in deep learning frameworks like:

* TensorFlow
* PyTorch

---

# Tensor Dimensions

| Tensor Type | Dimensions | Example                 |
| ----------- | ---------- | ----------------------- |
| Scalar      | 0D         | `5`                     |
| Vector      | 1D         | `[1,2,3]`               |
| Matrix      | 2D         | `[[1,2],[3,4]]`         |
| 3D Tensor   | 3D         | Multiple matrices       |
| nD Tensor   | nD         | Higher dimensional data |

---

# Examples of Tensors

---

## 0D Tensor (Scalar)

A single number.

```python
x = 5
```

Representation:

```text
5
```

---

## 1D Tensor (Vector)

```python
x = [1, 2, 3]
```

Representation:

```text
[1 2 3]
```

---

## 2D Tensor (Matrix)

```python
x = [
    [1, 2],
    [3, 4]
]
```

Representation:

```text
[
 [1 2]
 [3 4]
]
```

---

## 3D Tensor

A collection of matrices.

```python
x = [
      [[1,2], [3,4]],
      [[5,6], [7,8]]
    ]
```

Representation:

```text
[
  [
    [1 2]
    [3 4]
  ],

  [
    [5 6]
    [7 8]
  ]
]
```

---

# Easy Way to Understand

Think of dimensions step by step:

| Object          | Example           | Dimension |
| --------------- | ----------------- | --------- |
| Number          | `5`               | 0D        |
| List            | `[1,2,3]`         | 1D        |
| Table           | `[[1,2],[3,4]]`   | 2D        |
| Stack of tables | Multiple matrices | 3D        |

---

# Relationship Between Them

```text
Array → General data structure
Vector → 1D array
Tensor → Generalized multidimensional array
```

OR

```text
Scalar ⊂ Vector ⊂ Matrix ⊂ Tensor
```

---

# In Machine Learning and Deep Learning

## Arrays

Used in:

* NumPy
* Data storage
* Numerical computation

Example:

```python
import numpy as np

arr = np.array([1,2,3])
```

---

## Vectors

Used for:

* Feature representation
* Word embeddings
* Input features

Example:

```python
features = [height, weight, age]
```

---

## Tensors

Used in deep learning models.

Example in TensorFlow:

```python
import tensorflow as tf

tensor = tf.constant([[1,2],[3,4]])
```

Example in PyTorch:

```python
import torch

tensor = torch.tensor([[1,2],[3,4]])
```

---

# Difference Between Array, Vector, and Tensor

| Feature          | Array                  | Vector                | Tensor                                      |
| ---------------- | ---------------------- | --------------------- | ------------------------------------------- |
| Definition       | Collection of elements | 1D mathematical array | Multi-dimensional data structure            |
| Dimensions       | 1D, 2D, nD             | Mainly 1D             | 0D to nD                                    |
| Used In          | Programming            | Mathematics/ML        | Deep Learning                               |
| Example          | `[1,2,3]`              | `[1,2,3]`             | `[[1,2],[3,4]]`                             |
| Special Property | Data storage           | Magnitude + direction | Generalized multidimensional representation |

---

# Simple Real-World Analogy

| Object | Analogy                    |
| ------ | -------------------------- |
| Scalar | Single temperature reading |
| Vector | List of student marks      |
| Matrix | Excel sheet/table          |
| Tensor | Stack of Excel sheets      |

---

# Important Note

In practical deep learning:

* Every vector is a tensor
* Every matrix is a tensor
* But every tensor is not a vector

Because tensors can have many dimensions.
