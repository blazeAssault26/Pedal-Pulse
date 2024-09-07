# Project Overview

**Pedal Pulse** is a web-application, accessible at [Pedal Pulse Web App](https://pedal-pulse.streamlit.app/), that aims to predict the bike rental demand through **predictive analytics** and **statistical modeling**.

## Workflow

```
├── Business_Problem_Formulation
│   └── business_problem.md
├── Dataset
│   └── BikeData.csv
├── Know_Your_Data
│   └── variable_exploration
├── Data_Wrangling
│   ├── missing_values
│   └── skewness
│   ├── feature_creation
│   │   ├── Creation of new features (e.g., Day, Month, Year, Weekday) from the Date column.
│   │   └── Creation of Temperature Bin feature to aid in visualizing bike demand trends.
├── EDA_And_Data_Visualization
│   ├── univariate_analysis
│   └── bivariate_multivariate_analysis
├── Feature_Engineering
│   ├── vif_analysis
│   ├── removing_multicollinearity
│   └── encoding_categorical_variable
├── Model_Building
│   ├── train_test_split
│   ├── scaling_data
│   └── model_training
├── Model_Implementation
│   ├── linear_regression
│   ├── lasso
│   ├── ridge
│   ├── knn
│   ├── svm
│   ├── decision_tree
│   ├── random_forest
│   ├── xgboost
│   └── model_evaluation
├── Hyperparameter_Tuning
│   ├── random_forest_tuning
│   └── xgboost_tuning
├── Building_Web_App
│   ├── predictor.py
│   ├── app.py
│   └── model.py
├── requirements.txt
├── Dockerfile (docker pull anmol26/pedalpulse)
├── Deployment
│   └── streamlit_web_app

``` 

## Exploratory Data Analysis (EDA)

Through extensive **exploratory data analysis (EDA)**, this project provides valuable insights into demand patterns that will aid in formulating effective strategies for **fleet management**, **resource allocation**, and enhancing **customer satisfaction**.

<p float="left">
  <img src="https://github.com/blazeAssault26/Pedal-Pulse/assets/129224378/db1a3a51-7189-485c-9af0-0aad63726798" width="49%" />
  <img src="https://github.com/blazeAssault26/Pedal-Pulse/assets/129224378/9a953b70-ab7d-48f7-8aeb-ab8a6483f5b6" width="49%" />
</p>

## Model Implementation and Results

The project utilizes various **regression techniques** to model the bike demand data. Hyperparameters for **XGBoost** and **Random Forest** were tuned using **RandomSearchCV**, followed by **cross-validation** to rigorously evaluate their performance. This process optimized the evaluation metric, achieving an **R2 score** of **0.944** on the test data.

<p float="left">
  <img src="https://github.com/blazeAssault26/Pedal-Pulse/assets/129224378/55e079f4-4e39-4f8d-a097-6187ef7df938" width="49%" height="300" />
  <img src="https://github.com/blazeAssault26/Pedal-Pulse/assets/129224378/fe9602fa-be25-403a-9c81-7782edd1a3e5" width="49%" height="300" />
</p>
