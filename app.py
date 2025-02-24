import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
import joblib
import pickle

st.set_page_config(
    page_title="Diabetes Predictor",  # Title
    page_icon="🩺",    #💙           # Icon that will appear in the browser tab (can use text or image)
)
st.markdown("<h1 style='font-size: 100px;'>🏥</h1>", unsafe_allow_html=True)
st.title("Diabetes Predictor")
st.write("**Kindly provide your information to estimate your likelihood of having diabetes.**")

### HBA1C Predictor
## code used to train + evaluate hba1c model;
# hbdf = pd.read_csv("glucose.csv")
# hbdf = hbdf.dropna()    # get rid of null vals
# categorical_cols = hbdf.select_dtypes(include=['object']).columns
# encoder = LabelEncoder()
# for col in categorical_cols:
#     hbdf[col] = encoder.fit_transform(hbdf[col])

# X1 = hbdf.drop(columns=["HbA1c"])  # Features
# y1 = hbdf["HbA1c"]  # Target variable
# X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.15, random_state=42)

# Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X1_train, y1_train)

# joblib.dump(model, "hba1c_model.pkl")

##model evaluation
# Predictions
# y1_pred = model.predict(X1_test)

# Evaluate model
# mae = mean_absolute_error(y1_test, y1_pred)
# r2 = r2_score(y1_test, y1_pred)
# print(f"Mean Absolute Error: {mae}")
# print(f"R² Score: {r2}")  # Closer to 1 means better model fit

# finding and adjusting hyperparameters
# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [None, 10, 20]
# }

# grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
# grid_search.fit(X1_train, y1_train)
# print(grid_search.best_params_)

#loading model
try:
    model = joblib.load("hba1c_model.pkl")
    # model.fit(X1_train, y1_train)
    print("Model loaded successfully! It's a valid .pkl file.")
except Exception as e:
    print("Error loading file:", e)

# ## Diabetes Predictor
# # code used to training and evaluating the diabetes model
# df = pd.read_csv("diabetes_prediction_dataset.csv")
# # get rid of any null/missing values per column (smoking column -- get rid of "No Info")
# df = df.replace("No Info", pd.NA).dropna()
# df = df.replace("Other", pd.NA).dropna()

# # undersample majority class
# X = df.drop(columns=['diabetes']) 
# y = df['diabetes'] 
# undersample = RandomUnderSampler(sampling_strategy='auto', random_state=42)
# X_resampled, y_resampled = undersample.fit_resample(X, y)
# df = pd.DataFrame(X_resampled, columns=X.columns)
# df['diabetes'] = y_resampled 
# print(df.info())

# # check which columns are strings and turn them into ints
# categorical_cols = df.select_dtypes(include=['object']).columns
# encoder = LabelEncoder()
# for col in categorical_cols:
#     df[col] = encoder.fit_transform(df[col])
# X = df.drop(columns=['diabetes']) 
# y = df['diabetes'] 

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# diabetes_model = RandomForestRegressor(n_estimators=100, random_state=42)
# diabetes_model.fit(x_train, y_train)
# # Predictions
# y_pred = diabetes_model.predict(x_test)
# # Evaluate model
# d_mae = mean_absolute_error(y_test, y_pred)
# d_r2 = r2_score(y_test, y_pred)

# print("mean abs error:", d_mae)
# print("R&2 score:", d_r2)

# joblib.dump(diabetes_model, "diabetes_model.pkl")

try:
    diabetes_model = joblib.load("diabetes_model.pkl")
    # model.fit(X1_train, y1_train)
    print("Model loaded successfully! It's a valid .pkl file.")
except Exception as e:
    print("Error loading file:", e)

gender = st.radio("**Enter gender:**", [0, 1], format_func=lambda x: "Male" if x==1 else "Female", index=None)
age = st.number_input("**Enter Age**")
height = st.number_input("**Enter height(cm):**")
weight = st.number_input("**Enter weight(kg)**")
if height != 0:
  bmi = weight / ((height / 100) ** 2)
Waist_circumference = st.number_input("**Enter waist circumference (cm)**")
smoking_history = st.radio("**Enter Smoking History:**", [0, 1, 2, 3], format_func=lambda x: "Never" if x==3 else "Former" if x == 2 else "Not Current" if x == 1 else "Current", index=None )
hypertension = st.radio("**Do you have Hypertension?**", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=None)
heart_disease = st.radio("**Do you have Heart Disease?**", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", index=None)
HbA1c_lev = st.number_input("**Enter your A1c levels: (enter -1 if you do not know, we will ask you some questions to predict your A1c)**")
if HbA1c_lev < 0:
    Sex = gender
    Age = age
    Height = height
    Weight = weight
    HbA1c_cat = st.radio(
       """**Please select which A1c level you think you have based on these symptoms:**\n
**Symptoms of low A1c** -
Shakiness, Sweating, Dizziness, Confusion, Anxiety, Headache, Blurred vision, and Seizures


**Symptoms of high A1c** -
Increased thirst, Frequent urination, Excessive hunger, Blurred vision, Fatigue, Slow-healing wounds""",
       [1, 2, 3, 4],
       format_func=lambda x: "Normal" if x == 1 else "Boundary" if x == 2 else "Slightly High" if x == 3 else "High",
       index=None
    )
    Blood_pressure_cat = st.radio(
       "**Enter Blood Pressure:**",
       [1, 2, 3],
       format_func=lambda x: "Normal" if x == 1 else "Mild Risk" if x == 2 else "High Risk",
       index=None
    )
    HDL_cat = st.radio(
       "**Enter your HDL (commonly known as 'good' cholestrol) Level:**",
       [1, 2, 3],
       format_func=lambda x: "Normal" if x == 1 else "Slightly Low" if x == 2 else "Low",
       index=None
    )
    LDL_cat = st.radio(
       "**Enter your LDL (commonly known as 'bad' cholestrol) Level:**",
       [1, 2, 3],
       format_func=lambda x: "Normal" if x == 1 else "Slightly High" if x == 2 else "High",
        index=None
    )
  
    if smoking_history == 1:
        Smoking_status = 1
    else:
        Smoking_status = 0
    Hypertenstion_med = hypertension
    Hyperlipidemia_med = st.radio("**Enter if you take medication for Hyperlipidemia:**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=None)
    Cardiovascular_diseases = heart_disease
    Dialysis = st.radio("**Are you undergoing Dialysis?**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=None)
    Anemia_category = st.radio("**Do you have Anemia?**", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=None)
    Exercise30minutes_cat = st.radio("**Do you exercise for 30 minutes everyday?**",[1, 2], format_func=lambda x: "No" if x == 2 else "Yes", index=None)
    Alcohol_amount_cat = st.radio("**How much often do you consume alcohol?**", [1, 2, 3], format_func=lambda x: "Never" if x == 1 else "Sometimes" if x == 2 else "Everyday", index=None)

    HbA1c = None
    HbA1c_level = None
    prediction = model.predict([[Sex, Blood_pressure_cat, HDL_cat, LDL_cat, HbA1c_cat, Age, Height, Weight, Waist_circumference, bmi, Smoking_status, Hypertenstion_med, Hyperlipidemia_med, Cardiovascular_diseases, Dialysis, Anemia_category, Exercise30minutes_cat, Alcohol_amount_cat]])
    HbA1c = prediction[0]
    if HbA1c is not None:
        HbA1c_level = HbA1c
        if HbA1c_level is not None:
            st.write(f"**Predicted HbA1c level: {HbA1c_level:.2f}**")
    HbA1c_lev = HbA1c_level

    blood_glucose_level = st.radio("""**Enter your Blood Glucose Level Based on these Symptoms:**\n
**Symptoms of low glucose levels:**\n
Shakiness or trembling, Sweating, Hunger, Dizziness or lightheadedness, Confusion or difficulty concentrating, Headache, Anxiety or irritability\n
**Symptoms of high glucose levels:**\n
Urinating large amounts, Excessive thirst, Feeling tired, Frequent hunger, Dry mouth, Weight loss, Blurred vision""", [70, 110, 135.5, 170], format_func=lambda x: "Low" if x == 70 else 
            "Normal" if x == 110 else 
            "Slightly High" if x == 135.5 else "High" )


    # blood_glucose_level = st.number_input("**Enter your Blood Glucose Level:**")
    if st.button("**Calculate**"):
        if HbA1c_level is not None:
            prediction1 = diabetes_model.predict([[gender,age,bmi,smoking_history,hypertension,heart_disease,HbA1c_level,blood_glucose_level]])
            prediction1 = prediction1*100
            st.write(f"**Predicted Chance of Diabetes: {prediction1[0]:.2f}%**")
else:
    blood_glucose_level = st.radio("""**Enter your Blood Glucose Level Based on these Symptoms:**\n
**Symptoms of low glucose levels:**\n
Shakiness or trembling, Sweating, Hunger, Dizziness or lightheadedness, Confusion or difficulty concentrating, Headache, Anxiety or irritability\n
**Symptoms of high glucose levels:**\n
Urinating large amounts, Excessive thirst, Feeling tired, Frequent hunger, Dry mouth, Weight loss, Blurred vision""", [70, 110, 135.5, 170], format_func=lambda x: "Low" if x == 70 else 
            "Normal" if x == 110 else 
            "Slightly High" if x == 135.5 else "High" )


    
    # blood_glucose_level = st.number_input("**Enter your Blood Glucose Level:**")
    HbA1c_level = HbA1c_lev
    if st.button("**Calculate**"):
        prediction = diabetes_model.predict([[gender,age,bmi,smoking_history,hypertension,heart_disease,HbA1c_level,blood_glucose_level]])
        prediction = prediction*100
        st.write(f"**Predicted Chance of Diabetes: {prediction[0]:.2f}%**")