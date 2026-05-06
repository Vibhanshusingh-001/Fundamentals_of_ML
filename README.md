

# Difference Between Machine Learning and Deep Learning

## Difference between Machine Learning and Deep Learning:

| **Basis**            | **Machine Learning (ML)**                                      | **Deep Learning (DL)**                                      |
|---------------------|---------------------------------------------------------------|-------------------------------------------------------------|
| **Definition**      | Algorithms that learn from data and improve with experience.  | Subset of ML using multi-layered neural networks.           |
| **Data Requirement**| Works well with small to medium datasets.                    | Requires large datasets for effective learning.             |
| **Feature Extraction** | Manual – features must be selected by experts.            | Automatic – learns features directly from data.             |
| **Training Time**   | Faster and less resource-intensive.                          | Slower and needs more computation power.                    |
| **Accuracy**        | Depends on feature quality and algorithm.                    | Usually higher with enough data.                            |
| **Hardware Needs**  | Can run on CPUs.                                             | Often requires GPUs or TPUs.                                |
| **Interpretability**| Easy to interpret.                                           | Difficult to interpret (“black box”).                       |
| **Examples**        | Spam detection, stock prediction, recommendation systems.    | Image classification, speech recognition, NLP.              |
## Introduction

ML offers a new way to solve problems, answer complex questions, and create new content.  
ML can predict the weather, estimate travel times, recommend songs, auto-complete sentences, summarize articles, and generate never-seen-before images.

In basic terms, ML is the process of training a piece of software, called a model, to make useful predictions or generate content (like text, images, audio, or video) from data.



# Fundamentals_of_ML

## What is a "model" in machine learning?

A model is a mathematical relationship derived from data that an ML system uses to make predictions.



## Types of ML Systems

ML systems fall into one or more of the following categories based on how they learn to make predictions or generate content:

- Supervised learning  
- Unsupervised learning  
- Reinforcement learning  
- Generative AI  



## Supervised Learning

- CLAASIFICATION AND REGRESSION



## Unsupervised Learning

- Clustering
- Anomaly detection ----> Detect outliers
- Association
- Dimensionality readuction

## Semisupervised learning

- Do not have to label all data, label few points other will be automatically lablled e.g. Google Photos



## Reinforcement Learning--(no need of input data)

Reinforcement learning models make predictions by getting rewards or penalties based on actions performed within an environment.  
A reinforcement learning system generates a policy that defines the best strategy for getting the most rewards.

Reinforcement learning is used to train robots to perform tasks, like walking around a room, and software programs like AlphaGo to play the game of Go.

<img width="612" height="476" alt="image" src="https://github.com/user-attachments/assets/408f28c2-8e55-4be0-8146-c91b4ae8288a" />


## Generative AI

Generative AI is a class of models that creates content from user input.  
For example, generative AI can create unique images, music compositions, and jokes; it can summarize articles, explain how to perform a task, or edit a photo.

Generative AI can take a variety of inputs and create a variety of outputs, like text, images, audio, and video.  
It can also take and create combinations of these.

For example:
- A model can take an image as input and create an image and text as output  
- A model can take an image and text as input and create a video as output  


## Generative Model Input–Output Types

Generative models are often described using the format:  
**"type of input" → "type of output"**

Examples include:

- Text-to-text  
- Text-to-image  
- Text-to-video  
- Text-to-code  
- Text-to-speech  
- Image and text-to-image
<img width="1614" height="859" alt="Screenshot (791)" src="https://github.com/user-attachments/assets/a010e8df-88df-44a2-b8c0-722f6f35a33e" />

# Batch Machine Learning

**Batch Machine Learning** (also called **offline learning**) is a type of machine learning where the model is trained on the entire dataset at once, and then deployed. Once trained, the model does **not update continuously** with new data—it requires retraining from scratch or with accumulated data.



In batch learning, the system:

* Collects data
* Trains the model on all available data
* Deploys the model
* Updates only when retrained again


##  How It Works

1. Gather a dataset (historical data)
2. Train the model using that dataset
3. Save the trained model
4. Use the model for predictions
5. Periodically retrain with new data


##  Example

Suppose you are building a **spam email classifier**:

* You collect 10,000 labeled emails
* Train a model on this dataset
* Deploy it to classify new emails
* After a month, you collect new emails and retrain the model again



##  Characteristics

###  Advantages

* Simple to implement
* High accuracy (trained on full dataset)
* Stable (no frequent updates)

###  Disadvantages

* Cannot learn in real-time
* Requires retraining for new data
* Training can be slow for large datasets


##  Batch vs Online Learning

| Feature        | Batch Learning         | Online Learning               |
| -------------- | ---------------------- | ----------------------------- |
| Training Style | Entire dataset at once | Incremental (step-by-step)    |
| Data Handling  | Static dataset         | Continuous data stream        |
| Updates        | Periodic retraining    | Real-time updates             |
| Speed          | Slower training        | Faster updates                |
| Use Case       | Small/medium data      | Large-scale or streaming data |



##  When to Use Batch Learning

* When data is **not changing frequently**
* When you have **complete dataset available**
* When **high accuracy** is required
* When **real-time updates are not needed**


##  Real-world Applications

* Credit risk prediction
* Customer churn analysis
* Image classification (trained offline)
* Fraud detection (periodic updates)

**Online Learning** (also called **incremental learning**) is a machine learning approach where the model learns **continuously from incoming data**, updating itself step by step instead of training all at once.



##  Key Idea

Instead of waiting for a full dataset, the model:

* Learns from **one data point or small batches at a time**
* Updates its parameters **immediately**
* Adapts to **new patterns in real-time**



##  How It Works

1. Initialize the model
2. Receive a new data point (or small batch)
3. Make a prediction
4. Calculate error (loss)
5. Update model parameters
6. Repeat continuously



##  Example

Imagine a **stock price prediction system**:

* New stock data comes every second
* The model updates itself after each new data point
* It continuously improves based on the latest trends



##  Simple Mathematical View

At each step, parameters are updated using gradient descent:

w = w - \eta \nabla L(w)

* ( w ): model parameters
* ( \eta ): learning rate
* ( \nabla L(w) ): gradient of loss

 This update happens **frequently (per data point or mini-batch)** instead of after the whole dataset.



##  Characteristics

###  Advantages

* Learns in **real-time**
* Handles **large or streaming data**
* Adapts to **changing patterns (concept drift)**
* No need to store entire dataset

###  Disadvantages

* Can be **noisy and unstable**
* Sensitive to **outliers**
* May forget old patterns (**catastrophic forgetting**)
* Requires careful tuning (learning rate, etc.)



##  Online vs Batch Learning

| Feature        | Online Learning | Batch Learning |
| -------------- | --------------- | -------------- |
| Training Style | Incremental     | Entire dataset |
| Data Handling  | Streaming       | Static         |
| Updates        | Continuous      | Periodic       |
| Memory Usage   | Low             | High           |
| Adaptability   | High            | Low            |



##  When to Use Online Learning

* Real-time systems (recommendations, fraud detection)
* Data arrives continuously
* Dataset is too large to fit in memory
* Environment changes over time



##  Real-world Applications

* Recommendation systems (Netflix, YouTube)
* Fraud detection systems
* Ad click prediction
* Autonomous systems (self-driving updates)






