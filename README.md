# Customer Churn Prediction

## Project Overview
This project uses Machine Learning to predict whether a telecom customer is likely to leave a service provider. It uses customer account information, service details, and billing data to identify potential churn.

## Objectives
- Analyze customer data to identify churn patterns.
- Explore factors associated with customer churn.
- Train and evaluate machine learning models.
- Select a suitable model for churn prediction.
- Deploy the trained model using Streamlit.

## Dataset
- Dataset: IBM Telco Customer Churn
- Total records: 7,043
- Features: 20 input columns and 1 target column
- Target variable: Churn

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Google Colab

## Project Workflow
1. Load and explore the dataset.
2. Handle missing values and preprocess the data.
3. Perform Exploratory Data Analysis (EDA).
4. Encode categorical features and scale numerical features.
5. Split the dataset into training and testing sets.
6. Train Logistic Regression and Random Forest models.
7. Evaluate model performance.
8. Tune the classification threshold.
9. Save the trained model.
10. Deploy the model using Streamlit.

## Model Performance

### Logistic Regression
- Accuracy: 73.81%
- ROC-AUC: 0.841
- Churn precision: 0.50
- Churn recall: 0.78
- Churn F1-score: 0.61

### Random Forest
- Accuracy: 78.21%
- ROC-AUC: Approximately 0.822
- Churn precision: 0.62
- Churn recall: 0.47
- Churn F1-score: 0.53

### Final Model
Logistic Regression was selected because it achieved better churn recall and ROC-AUC.

A classification threshold of 0.45 was selected to balance precision and recall.

At this threshold:
- Churn precision: Approximately 48%
- Churn recall: Approximately 84%
- Churn F1-score: Approximately 61%

## Deployment
The trained model is deployed through a Streamlit web application.

The application accepts customer information and displays:
- Predicted churn probability
- Churn prediction
- Prediction threshold

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt