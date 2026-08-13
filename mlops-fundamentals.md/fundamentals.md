# MLOps Fundamentals: From Machine Learning Model to Production

## Introduction

Machine Learning is no longer limited to notebooks and research environments. Businesses increasingly use ML models for **fraud detection, recommendation systems, demand forecasting, customer churn prediction, credit scoring, predictive maintenance, and personalization**.

However, building an accurate model is only one part of the problem.

A model that performs well in a Jupyter Notebook can still fail in production because:

* Production data changes.
* Feature distributions change.
* Model accuracy degrades.
* Training cannot be reproduced.
* Dependencies become incompatible.
* Manual deployment becomes error-prone.
* Infrastructure becomes expensive.
* There is no proper model versioning.
* Engineers cannot easily roll back a bad model.
* Nobody knows when the model has started producing poor predictions.

This is where **MLOps — Machine Learning Operations** comes in.

MLOps applies software engineering, DevOps, automation, infrastructure, monitoring, and governance practices to the **entire machine learning lifecycle**.

Google describes MLOps as an engineering culture and practice that unifies ML development and operations, with automation and monitoring across integration, testing, releasing, deployment, and infrastructure management.

---

# 1. What is MLOps?

**MLOps = Machine Learning + Development + Operations**

A simple definition is:

> **MLOps is the discipline of building, deploying, monitoring, maintaining, and continuously improving machine learning systems in production.**

Traditional software engineering generally looks like:

```text
Code
  ↓
Build
  ↓
Test
  ↓
Deploy
  ↓
Monitor
```

ML systems are more complicated:

```text
Data
  ↓
Data Validation
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Model Registry
  ↓
Deployment
  ↓
Inference
  ↓
Monitoring
  ↓
New Data
  ↓
Retraining
  ↓
New Model
```

The important difference is that **the production artifact is not just code**.

An ML system depends on:

* Code
* Data
* Features
* Model
* Model parameters
* Dependencies
* Infrastructure
* Configuration
* Training pipeline
* Serving infrastructure
* Monitoring

This is why MLOps is more complex than simply applying CI/CD to Python code.

---

# 2. Why Do We Need MLOps?

Imagine a company creates a fraud detection model.

During development:

```text
Accuracy = 97%
Precision = 95%
Recall = 93%
```

The data scientist deploys it.

Six months later:

```text
Accuracy = 82%
Precision = 76%
Recall = 70%
```

But the application itself is still running.

There might be:

```text
No HTTP error
No Kubernetes failure
No application crash
No CPU alert
No memory alert
```

The infrastructure looks healthy.

The **ML system is failing silently**.

This is one of the fundamental challenges of production ML systems: models can degrade because the underlying data or real-world environment changes without producing conventional application errors.

MLOps provides mechanisms to detect and respond to these problems.

---

# 3. MLOps vs DevOps

DevOps focuses primarily on software systems.

MLOps extends those principles to ML-specific artifacts and processes.

| DevOps                    | MLOps                              |
| ------------------------- | ---------------------------------- |
| Source code               | Source code + ML code              |
| Application testing       | Application + data + model testing |
| CI/CD                     | CI/CD + Continuous Training        |
| Application deployment    | Model + application deployment     |
| Infrastructure monitoring | Infrastructure + model monitoring  |
| Software versioning       | Code + data + model versioning     |
| Release management        | Model promotion and release        |
| Rollback application      | Rollback model                     |
| Logs                      | Logs + prediction/model metrics    |
| Infrastructure health     | Infrastructure + data/model health |

MLOps therefore requires CI/CD practices plus ML-specific capabilities such as **data validation, model validation, continuous training, model management, and model performance monitoring**.

---

# 4. The MLOps Lifecycle

A practical MLOps lifecycle can be represented as:

```text
                 ┌──────────────────┐
                 │  Business Problem │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │    Data          │
                 │ Collection       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Data Validation  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Feature          │
                 │ Engineering      │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Training   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Evaluation │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Registry   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Deployment       │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Inference        │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Monitoring       │
                 └────────┬─────────┘
                          ↓
                    Drift / New Data
                          │
                          └──────────→ Retraining
```

The lifecycle is continuous rather than a one-time deployment process. Production MLOps commonly includes continuous training, model serving, monitoring, dataset/feature management, and model governance.

---

# 5. Data Management

ML starts with data.

An MLOps Engineer needs to make sure that training and production data are:

* Available
* Reliable
* Versioned
* Validated
* Reproducible
* Secure
* Traceable

For example:

```text
customer_id
age
income
transaction_count
average_transaction
fraud_label
```

A production pipeline might receive:

```text
age = "unknown"
income = null
transaction_count = -10
```

A model may still technically execute, but its prediction quality can become unreliable.

Therefore, data validation should happen before training or inference.

Common concepts include:

* Data quality
* Schema validation
* Missing-value detection
* Outlier detection
* Distribution checks
* Data lineage
* Dataset versioning

Example tools:

```text
DVC
LakeFS
Great Expectations
AWS S3
Google Cloud Storage
Azure Blob Storage
Delta Lake
```

---

# 6. Feature Engineering

Features are the inputs used by the model.

For a customer churn model:

```text
monthly_usage
number_of_complaints
contract_duration
payment_history
average_monthly_bill
```

The feature pipeline might look like:

```text
Raw Data
   ↓
Cleaning
   ↓
Transformation
   ↓
Feature Engineering
   ↓
Feature Store
   ↓
Training / Inference
```

One important production problem is **training-serving skew**.

For example:

Training calculates:

```text
average_transaction = total_transactions / 30
```

Production calculates:

```text
average_transaction = total_transactions / 7
```

The model receives different feature definitions between training and production.

This can significantly affect predictions.

---

# 7. Experiment Tracking

Data scientists usually run many experiments.

For example:

```text
Experiment 1
Random Forest
Accuracy = 91%

Experiment 2
XGBoost
Accuracy = 94%

Experiment 3
LightGBM
Accuracy = 95%

Experiment 4
XGBoost + feature engineering
Accuracy = 97%
```

Without experiment tracking, it becomes difficult to answer:

> Which dataset, code, parameters, and model produced this result?

An experiment tracking system records things such as:

```text
Model
Dataset version
Hyperparameters
Metrics
Code version
Artifacts
Training time
```

A common tool is:

**MLflow**

Example:

```text
Experiment
    │
    ├── Run 1
    │     ├── Parameters
    │     ├── Metrics
    │     └── Model
    │
    ├── Run 2
    │     ├── Parameters
    │     ├── Metrics
    │     └── Model
    │
    └── Run 3
          ├── Parameters
          ├── Metrics
          └── Model
```

---

# 8. Model Versioning

Suppose production currently uses:

```text
fraud-model:v10
```

A new model is trained:

```text
fraud-model:v11
```

Before deployment, the MLOps pipeline should evaluate:

```text
v10 vs v11
```

For example:

| Metric    |    v10 |    v11 |
| --------- | -----: | -----: |
| Accuracy  |    94% |    96% |
| Precision |    91% |    94% |
| Recall    |    88% |    92% |
| Latency   | 120 ms | 125 ms |

If v11 satisfies the release criteria, it can be promoted.

```text
Training
   ↓
Evaluation
   ↓
Candidate Model
   ↓
Model Registry
   ↓
Staging
   ↓
Production
```

This provides reproducibility and controlled model promotion.

---

# 9. Model Registry

A **Model Registry** manages models throughout their lifecycle.

Example:

```text
fraud-model
│
├── Version 1
├── Version 2
├── Version 3
├── Version 4
└── Version 5
```

A model may have states such as:

```text
Candidate
   ↓
Staging
   ↓
Production
   ↓
Archived
```

The registry should help answer:

* Which model is in production?
* Who trained it?
* Which dataset was used?
* What metrics did it achieve?
* When was it deployed?
* Which version should be rolled back?

---

# 10. CI for Machine Learning

Traditional CI tests application code.

MLOps CI should test more than code.

A pipeline might look like:

```text
Git Push
   ↓
Run Unit Tests
   ↓
Run Data Validation
   ↓
Run Feature Tests
   ↓
Train/Test Model
   ↓
Evaluate Metrics
   ↓
Build Container
   ↓
Security Scan
```

Example quality gates:

```text
Unit tests       → PASS
Data validation  → PASS
Model accuracy   → >= 95%
Precision        → >= 90%
Security scan    → PASS
Container build  → PASS
```

Only if all gates pass should the pipeline continue.

---

# 11. CD for Machine Learning

Continuous Delivery automates the delivery of the ML system.

A simplified pipeline:

```text
GitHub
   ↓
CI
   ↓
Build Docker Image
   ↓
Push Image
   ↓
Deploy to Staging
   ↓
Integration Tests
   ↓
Model Validation
   ↓
Production
```

Common tools include:

```text
GitHub Actions
Jenkins
GitLab CI/CD
Argo CD
Docker
Kubernetes
Helm
Terraform
```

---

# 12. Continuous Training — CT

One of the major differences between DevOps and MLOps is **Continuous Training**.

Traditional software:

```text
Developer changes code
       ↓
CI/CD
       ↓
Deployment
```

ML:

```text
New Data
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Registry
   ↓
Deployment
```

A continuous training pipeline can automatically retrain a model when:

* New training data arrives.
* Model performance drops.
* Data drift is detected.
* A scheduled training window occurs.

Google's MLOps guidance explicitly describes continuous training as an important capability for automatically retraining candidate models and validating them before serving.

---

# 13. Model Deployment

There are several ways to serve ML models.

## Batch Inference

Predictions are generated periodically.

```text
Database
   ↓
Batch Job
   ↓
ML Model
   ↓
Predictions
   ↓
Database
```

Example:

A bank predicts customer churn every night.

---

## Real-Time Inference

The application sends a request to the model.

```text
Application
     ↓
API
     ↓
Model Server
     ↓
Prediction
```

Example:

```http
POST /predict
```

Request:

```json
{
  "age": 35,
  "income": 65000,
  "transactions": 42
}
```

Response:

```json
{
  "fraud_probability": 0.91,
  "prediction": "fraud"
}
```

Common serving technologies include:

```text
FastAPI
BentoML
KServe
TorchServe
TensorFlow Serving
NVIDIA Triton
```

---

# 14. Docker for MLOps

A model should not depend on the developer's laptop.

Instead, package the application into a container.

```text
Python
+
Dependencies
+
Model
+
Application
+
Configuration
        ↓
Docker Image
```

Example:

```text
fraud-api:1.0
```

This image can run consistently across:

```text
Developer laptop
        ↓
CI
        ↓
Staging
        ↓
Kubernetes
        ↓
Production
```

---

# 15. Kubernetes for MLOps

Kubernetes becomes useful when ML services need:

* Scalability
* High availability
* Service discovery
* Resource management
* Rolling deployments
* GPU scheduling
* Self-healing

Example:

```text
                 Load Balancer
                       ↓
                Kubernetes Service
                       ↓
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
    Model Pod      Model Pod      Model Pod
        ↓              ↓              ↓
     Model          Model          Model
```

If traffic increases:

```text
3 replicas
   ↓
10 replicas
```

Kubernetes can scale the inference service according to the workload.

---

# 16. Model Monitoring

Monitoring is one of the most important MLOps responsibilities.

Traditional monitoring asks:

```text
Is the service up?
```

MLOps also asks:

```text
Is the model still performing correctly?
```

Monitor multiple dimensions.

### Infrastructure Metrics

```text
CPU
Memory
GPU
Disk
Network
Pod health
```

### Application Metrics

```text
Request count
Error rate
Latency
Throughput
```

### ML Metrics

```text
Accuracy
Precision
Recall
F1
AUC
Prediction distribution
```

### Data Metrics

```text
Missing values
Feature distribution
Outliers
Schema changes
Data drift
```

### Business Metrics

```text
Fraud prevented
Revenue
Customer churn
Conversion rate
False positives
Cost savings
```

MLOps monitoring therefore needs to connect technical model metrics with production behavior and, ideally, business outcomes.

---

# 17. Data Drift

Data drift occurs when production input data changes compared with the data used during training.

Suppose a model was trained on:

```text
Average customer age = 35
```

Production becomes:

```text
Average customer age = 52
```

The distribution has changed.

Conceptually:

```text
Training Data Distribution
████████████████████

Production Distribution
        ███████████████████
```

This may indicate that the model needs investigation or retraining.

---

# 18. Concept Drift

Concept drift is different.

The relationship between inputs and the target changes.

For example:

```text
Before:
High transaction frequency → high fraud probability
```

After a major change in customer behavior:

```text
High transaction frequency → normal customer behavior
```

The input data may look similar, but the relationship between features and outcomes has changed.

This is why production ML requires ongoing evaluation.

---

# 19. Model Performance Monitoring

Suppose:

```text
Production Model

Accuracy
Day 1  → 96%
Day 7  → 95%
Day 14 → 93%
Day 21 → 90%
Day 30 → 84%
```

A monitoring system should detect this degradation.

Example alert:

```text
ALERT

Model: fraud-model:v12

Current Accuracy: 84%
Threshold: 90%

Action:
Trigger model investigation/retraining.
```

---

# 20. MLOps Architecture

A practical enterprise architecture could look like:

```text
                    Developer
                       │
                       ↓
                    GitHub
                       │
                       ↓
                CI/CD Pipeline
                       │
            ┌──────────┴──────────┐
            ↓                     ↓
       Data Pipeline          ML Training
            ↓                     ↓
       Data Validation       Experiment Tracking
            │                     │
            └──────────┬──────────┘
                       ↓
                  Model Registry
                       │
                       ↓
                  Model Validation
                       │
                       ↓
                 Docker Image
                       │
                       ↓
                  Kubernetes
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
        Model Service        Batch Jobs
             │                   │
             └─────────┬─────────┘
                       ↓
                  Predictions
                       │
                       ↓
                  Monitoring
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Metrics      Data Drift   Model Drift
          │            │            │
          └────────────┴────────────┘
                       ↓
                  Retraining
                       │
                       └────────→ Model Registry
```

---

# 21. Business Scenario: E-Commerce Customer Churn Prediction

Now let's understand MLOps from the perspective of an **MLOps Engineer**.

## Business Problem

An e-commerce company has:

```text
10 million customers
```

The company discovers that many customers stop purchasing after several months.

The business wants to answer:

> Which customers are likely to stop purchasing within the next 30 days?

The business objective is to reduce customer churn.

---

# 22. Business Solution

The Data Science team develops a churn prediction model.

Input features:

```text
customer_age
purchase_frequency
average_order_value
days_since_last_purchase
number_of_complaints
discount_usage
website_visits
```

Output:

```text
Churn Probability
```

Example:

```text
Customer A → 0.92
Customer B → 0.14
Customer C → 0.81
```

The business can target customers with high churn probability.

---

# 23. Where Does the MLOps Engineer Come In?

The Data Scientist creates the model.

The MLOps Engineer makes the model **production-ready, repeatable, observable, and maintainable**.

The responsibilities might look like:

```text
Data
 ↓
Training Pipeline
 ↓
Experiment Tracking
 ↓
Model Registry
 ↓
CI/CD
 ↓
Docker
 ↓
Kubernetes
 ↓
Model API
 ↓
Monitoring
 ↓
Drift Detection
 ↓
Retraining
```

---

# 24. Step-by-Step Business Workflow

## Step 1 — Data Collection

Customer data is collected from:

```text
PostgreSQL
Data Warehouse
S3
Kafka
CRM
Application databases
```

Example:

```text
customer_id = 10001
purchase_count = 3
last_purchase_days = 45
average_order_value = 1200
complaints = 4
```

---

## Step 2 — Data Validation

The MLOps pipeline checks:

```text
Schema
Missing values
Duplicate records
Invalid values
Data types
Distribution
```

Example:

```text
purchase_count < 0
```

should fail validation.

---

## Step 3 — Feature Engineering

Create features:

```text
purchase_frequency
customer_lifetime_value
days_since_last_purchase
complaint_ratio
discount_dependency
```

---

## Step 4 — Model Training

The pipeline trains:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
```

Example result:

```text
Model       AUC
----------------
Logistic    0.81
RandomForest 0.87
XGBoost     0.92
LightGBM    0.91
```

The pipeline selects XGBoost.

---

# 25. Step 5 — Experiment Tracking

MLflow records:

```text
Experiment: customer-churn

Model:
XGBoost

Dataset:
dataset:v17

Parameters:
learning_rate=0.05
max_depth=6
n_estimators=300

Metrics:
AUC=0.92
Precision=0.89
Recall=0.86
```

---

# 26. Step 6 — Model Registry

Register:

```text
customer-churn-model:v21
```

Lifecycle:

```text
Candidate
   ↓
Staging
   ↓
Production
```

---

# 27. Step 7 — CI/CD

Developer changes the training code.

GitHub receives the commit:

```text
git push
```

Pipeline:

```text
GitHub
  ↓
Unit Tests
  ↓
Data Tests
  ↓
Training
  ↓
Model Evaluation
  ↓
Docker Build
  ↓
Security Scan
  ↓
Model Registry
  ↓
Deployment
```

---

# 28. Step 8 — Deploy the Model

The model is packaged:

```text
customer-churn-api:1.5
```

and deployed to Kubernetes.

```text
                  Load Balancer
                       ↓
                Churn API Service
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        Pod 1        Pod 2        Pod 3
```

The application sends:

```json
{
  "purchase_frequency": 2,
  "days_since_last_purchase": 60,
  "complaints": 3,
  "average_order_value": 900
}
```

The model returns:

```json
{
  "churn_probability": 0.87,
  "risk": "HIGH"
}
```

---

# 29. Step 9 — Business Action

The CRM system receives:

```text
Customer ID: 10001
Churn Probability: 87%
Risk: HIGH
```

The business automatically sends:

```text
Personalized Offer
Discount
Email
Push Notification
Loyalty Campaign
```

The ML model therefore directly influences business operations.

---

# 30. Step 10 — Monitor the Model

The MLOps Engineer monitors:

```text
API latency
Error rate
CPU
Memory
Prediction volume
Prediction distribution
Feature drift
Model accuracy
Business conversion
```

Example:

```text
API latency       = 80 ms
Error rate        = 0.2%
Prediction volume = 1.5M/day
Model AUC         = 0.91
```

Everything looks healthy.

---

# 31. Step 11 — Detect Drift

Three months later:

```text
Model AUC

Month 1 → 0.92
Month 2 → 0.91
Month 3 → 0.87
Month 4 → 0.81
```

The MLOps system detects degradation.

At the same time:

```text
Customer behavior changed
```

For example, the company introduced a new loyalty program.

Therefore, customer purchasing behavior changed.

---

# 32. Step 12 — Retraining

The MLOps pipeline automatically starts:

```text
New Data
   ↓
Validation
   ↓
Feature Engineering
   ↓
Training
   ↓
Evaluation
   ↓
Comparison
```

Old model:

```text
v21
AUC = 0.81
```

New model:

```text
v22
AUC = 0.91
```

The pipeline promotes:

```text
v22 → Production
```

This creates a continuous ML lifecycle.

---

# 33. What Does the MLOps Engineer Actually Own?

In this scenario, the MLOps Engineer may own:

### Infrastructure

```text
AWS
Kubernetes
Terraform
Docker
S3
EKS
IAM
```

### CI/CD

```text
GitHub Actions
Jenkins
Argo CD
```

### ML Lifecycle

```text
MLflow
DVC
Model Registry
Training pipelines
Model validation
```

### Orchestration

```text
Kubeflow
Airflow
Argo Workflows
Kubernetes Jobs
```

### Serving

```text
FastAPI
BentoML
KServe
```

### Observability

```text
Prometheus
Grafana
Loki
ELK
```

### Security

```text
Trivy
Gitleaks
IAM
Secrets Management
RBAC
```

---

# 34. MLOps Toolchain

A practical MLOps stack could be:

| Layer               | Tools                             |
| ------------------- | --------------------------------- |
| Programming         | Python                            |
| Source Control      | Git, GitHub                       |
| Data Versioning     | DVC, LakeFS                       |
| Experiment Tracking | MLflow                            |
| Data Validation     | Great Expectations                |
| Training            | Scikit-learn, PyTorch, TensorFlow |
| Pipeline            | Kubeflow, Airflow                 |
| Containerization    | Docker                            |
| CI/CD               | GitHub Actions, Jenkins           |
| Model Registry      | MLflow                            |
| Serving             | FastAPI, KServe, BentoML          |
| Orchestration       | Kubernetes                        |
| Cloud               | AWS, Azure, GCP                   |
| IaC                 | Terraform                         |
| Monitoring          | Prometheus, Grafana               |
| Logging             | Loki, ELK                         |
| Security            | Trivy, Gitleaks                   |
| GitOps              | Argo CD                           |

The exact stack varies by organization; MLOps is a set of practices rather than a requirement to use one particular vendor or tool.

---

# 35. MLOps Maturity Levels

A useful way to think about MLOps maturity is:

## Level 0 — Manual

```text
Data Scientist
    ↓
Notebook
    ↓
Train Model
    ↓
Manual Deployment
```

Problems:

* Manual processes
* Poor reproducibility
* Difficult deployment
* Limited monitoring

---

## Level 1 — Pipeline Automation

```text
Data
 ↓
Automated Training
 ↓
Validation
 ↓
Model
```

Training and validation become automated.

---

## Level 2 — CI/CD + Continuous Training

```text
Git
 ↓
CI
 ↓
Training Pipeline
 ↓
Model Validation
 ↓
CD
 ↓
Production
```

Models can be continuously retrained and delivered.

Google's MLOps maturity guidance similarly distinguishes manual processes from automated ML pipelines and more advanced CI/CD-oriented ML systems.

---

# 36. MLOps Engineer vs ML Engineer

There is overlap, but the focus differs.

| ML Engineer         | MLOps Engineer           |
| ------------------- | ------------------------ |
| Model development   | Model operationalization |
| Feature engineering | Pipeline automation      |
| Model optimization  | Deployment automation    |
| ML algorithms       | Infrastructure           |
| Training            | CI/CD                    |
| Model architecture  | Kubernetes               |
| Prediction quality  | Observability            |
| Model serving       | Production reliability   |

In many organizations, these responsibilities overlap significantly.

---

# 37. MLOps Engineer's Core Objective

The ultimate goal is not:

> "Deploy a model."

The goal is:

> **Build a reliable system that can repeatedly train, validate, deploy, monitor, and improve ML models in production.**

A mature MLOps platform should make this workflow almost automatic:

```text
New Data
   ↓
Validate
   ↓
Train
   ↓
Evaluate
   ↓
Register
   ↓
Deploy
   ↓
Monitor
   ↓
Detect Drift
   ↓
Retrain
   ↓
Evaluate
   ↓
Deploy Better Model
```

---

# 38. Business Metrics Matter

An MLOps Engineer should not focus only on:

```text
CPU
Memory
Latency
Accuracy
```

They should understand business impact.

For the churn example:

```text
Model Accuracy
       ↓
Churn Identification
       ↓
Customer Retention Campaign
       ↓
More Retained Customers
       ↓
Higher Revenue
```

Therefore, useful business KPIs could include:

```text
Customer retention rate
Revenue saved
Campaign conversion
Cost per retained customer
False-positive campaign cost
Customer lifetime value
```

This is an important mindset shift:

**MLOps is not just infrastructure for ML. It is infrastructure that enables ML to produce reliable business value.**

---

# 39. Key MLOps Principles

Remember these principles:

### 1. Automate

Automate repetitive processes.

### 2. Reproduce

A model should be reproducible from known inputs and versions.

### 3. Version

Version:

```text
Code
Data
Features
Models
Configuration
Infrastructure
```

### 4. Test

Test:

```text
Code
Data
Features
Models
Pipelines
Infrastructure
```

### 5. Monitor

Monitor:

```text
Infrastructure
Application
Data
Model
Business
```

### 6. Govern

Track:

```text
Who trained the model?
Which data was used?
Which version is deployed?
When was it deployed?
What changed?
```

### 7. Continuously Improve

Production feedback should feed the next iteration.

---

# 40. Final MLOps Mental Model

If you are preparing for an **MLOps Engineer** role, think about the entire lifecycle:

```text
                 BUSINESS
                    │
                    ↓
               ML PROBLEM
                    │
                    ↓
                  DATA
                    │
                    ↓
             DATA PIPELINE
                    │
                    ↓
             FEATURE ENGINEERING
                    │
                    ↓
              MODEL TRAINING
                    │
                    ↓
            EXPERIMENT TRACKING
                    │
                    ↓
              MODEL EVALUATION
                    │
                    ↓
               MODEL REGISTRY
                    │
                    ↓
              CI/CD PIPELINE
                    │
                    ↓
             DOCKER CONTAINER
                    │
                    ↓
                KUBERNETES
                    │
                    ↓
              MODEL SERVING
                    │
                    ↓
               PRODUCTION
                    │
                    ↓
               MONITORING
                    │
           ┌────────┴────────┐
           ↓                 ↓
       DATA DRIFT        MODEL DRIFT
           │                 │
           └────────┬────────┘
                    ↓
               RETRAINING
                    │
                    └────────→ NEW MODEL
```

## Conclusion

MLOps is the engineering discipline that turns an ML model into a **reliable production system**.

A Data Scientist may answer:

> **"Can we build a model that predicts customer churn?"**

An MLOps Engineer asks:

> **"Can we reliably train, validate, version, deploy, monitor, scale, secure, and continuously improve that model in production?"**

That distinction is the foundation of MLOps.

For an MLOps Engineer, the strongest skillset combines:

```text
Python
+
Machine Learning Fundamentals
+
Linux
+
Git
+
Docker
+
CI/CD
+
Cloud
+
Kubernetes
+
Terraform
+
MLflow
+
Data/Model Versioning
+
ML Pipelines
+
Model Serving
+
Observability
+
Security
```

The most important practical objective is to build an **end-to-end MLOps project** where you can demonstrate the complete journey from business problem → data → training → model registry → CI/CD → deployment → monitoring → drift → retraining.

### Further Reading

* [Google Cloud — MLOps: Continuous Delivery and Automation Pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)
* [Google Cloud — Practitioner's Guide to MLOps](https://cloud.google.com/resources/mlops-whitepaper?utm_source=chatgpt.com)
* [Google Cloud — MLOps Foundation](https://cloud.google.com/blog/products/ai-machine-learning/key-requirements-for-an-mlops-foundation?utm_source=chatgpt.com)
