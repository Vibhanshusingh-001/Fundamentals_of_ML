

## **Loss in Machine Learning**

### **What is Loss?**

Loss is a **numerical metric** that describes how incorrect a model’s predictions are.
It measures the **difference between the model’s predicted values and the actual (true) labels**.

### **Purpose of Loss**

The primary goal of training a machine learning model is to **minimize the loss**, reducing it to the **lowest possible value**.
A lower loss indicates better model performance.

### **Loss in Statistics and Machine Learning**

In both statistics and machine learning, loss quantifies the **distance between predicted and actual values**.

* Loss focuses on **how far apart the values are**
* It does **not** focus on the direction of the error (positive or negative)

#### **Example**

If a model predicts **2** and the actual value is **5**:

* The raw difference is: `2 − 5 = −3`
* The sign (−) does not matter
* What matters is the **distance between the values**, which is **3**

Because of this, loss functions always **remove the sign** of the difference.

### **Common Methods to Remove the Sign**

The two most common approaches are:

1. **Absolute Error**
   Take the absolute value of the difference between the actual value and the prediction.

2. **Squared Error**
   Square the difference between the actual value and the prediction.
<img width="1192" height="748" alt="image" src="https://github.com/user-attachments/assets/c56666e7-9f4b-4f39-bda8-6ba54bac0bd5" />
