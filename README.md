# 🎓 Student Performance Prediction System

## 🚀 Overview

This project delivers a **production-ready machine learning system** to predict student exam performance based on demographic and academic attributes. It incorporates robust **MLOps practices**, including data pipelines, model training, Dockerized Flask deployment, and a CI/CD pipeline integrated with **AWS**.

> 🎯 **Goal:** Help educational institutions identify and support students at academic risk through real-time, data-driven predictions.

---

## 🧩 Problem Statement

### 🎓 Business Context

Educational institutions often struggle to identify underperforming students in time. Manual tracking is:
- Time-consuming  
- Prone to human bias  
- Often reactive rather than proactive

This system offers **automated early predictions** of academic performance using structured data, empowering educators to take timely and personalized actions.

### 📊 Machine Learning Objective

We treat this as a **regression problem** — predicting a student's final exam score based on:
- **Demographic features:** Gender, Parental Education, Lunch Type, Test Preparation Status  
- **Academic scores:** Math, Reading, and Writing

---

## 🧪 Dataset

**Source:** [Kaggle - Student Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

| Feature Type | Features |
|--------------|----------|
| Categorical  | Gender, Parental Education, Lunch Type, Test Preparation |
| Numerical    | Math Score, Reading Score, Writing Score                |

---

## 🧱 Project Architecture

This project follows a modular structure with 3 core components:

### 📁 1. Data Pipeline
**Purpose:** Prepare data for training  
- Data loading and inspection  
- Handle missing values  
- One-hot encoding of categorical variables  
- Normalize numerical features  
- Train-test split  

✅ *Output: Cleaned and transformed dataset*

---

### 🤖 2. Model Training Pipeline
**Purpose:** Build and evaluate ML models  
- Algorithms: Linear Regression, Random Forest, XGBoost  
- Evaluation metrics: MAE, R² Score  
- Save the best model using `joblib`  

✅ *Output: Serialized model for deployment*

---

### 🌐 3. Deployment & CI/CD

#### 🔧 Flask Web App
- User-friendly UI to input student data  
- Returns real-time score predictions  
- Built with Flask and Bootstrap

#### ☁️ Deployment Stack
- Dockerized web app  
- Hosted on **AWS Elastic Beanstalk**  
- Docker images stored on **AWS ECR**

#### 🔁 CI/CD Pipeline
- Managed with **GitHub Actions**  
- Automates:
  - Code linting and testing  
  - Docker image build and push  
  - Deployment to AWS

✅ *Output: Scalable and production-ready ML application*

---

## 📷 Application Demo *(Optional)*
- Responsive UI (desktop/mobile)  
- Instant score prediction after form submission  
- Lightweight design for educators and school administrators

---

## 🧠 Key Features

| Feature                 | Description                                     |
|-------------------------|-------------------------------------------------|
| ✅ End-to-End Pipeline  | From data ingestion to real-time prediction     |
| 🔍 Model Evaluation     | Performance measured using MAE and R²           |
| 🌐 Web Interface        | Flask-based app for score prediction            |
| 📦 Dockerized System    | Portable and consistent across environments     |
| 🚀 CI/CD Automation     | Seamless deployment using GitHub Actions & AWS  |
| ☁️ Cloud Deployment     | Hosted on AWS Elastic Beanstalk (Production)    |

---

## 🎯 Project Impact

This system enables:
- Early detection of students at risk of underperforming  
- Timely academic support through actionable insights  
- Scalable monitoring powered by data rather than guesswork

---

## 📂 Tech Stack

- **Languages:** Python  
- **Frameworks:** Flask, scikit-learn  
- **Libraries:** pandas, NumPy, seaborn, matplotlib, joblib  
- **MLOps Tools:** Docker, GitHub Actions, AWS ECR, EC2, Elastic Beanstalk

---

## 🤝 Connect With Me

If you'd like to contribute, collaborate, or have questions:

- 📧 Email: [nagarn603@gmail.com](mailto:nagarn603@gmail.com)  
- 🌐 LinkedIn: [Nikhil Nagar](https://www.linkedin.com/in/nikhil-nagar-996204264/)

---

