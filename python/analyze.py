from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
import joblib
import PIL.Image
import os
import google.generativeai as genai
<<<<<<< HEAD
# from IPython.display import display
=======
genai.configure(api_key="AIzaSyDvwICRzk-Replace your own gemini api key")
>>>>>>> 49e07b6aa74aa4e7dcdaf5075785ed7392ce61b5

genai.configure(api_key="AIzaSyBHqbr3O6DM7snkq3fwa6ZB2uQPj6ZqQs0")

#img = PIL.Image.open('blood2.png')

# Locate the first image in the uploads folder

uploads_dir = 'uploads'
image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise FileNotFoundError("No image files found in the uploads folder.")
first_image_path = os.path.join(uploads_dir, image_files[0])

# Load the image
img = PIL.Image.open(first_image_path)

model = genai.GenerativeModel('gemini-2.5-flash')

text_prompt = """
extract these medical terms actual numeric value without unit(unit lakhs should be in numeric form) from the image if available
and must ignore the missing values.
medical_terms = {
    'Glucose': ['Glucose','Glu','G','g'],
    'Cholesterol': ['Cholesterol','Chol','C','c'],
    'Hemoglobin': ['Hemoglobin','Haemoglobin', 'HEMOGLOBIN','Hb','Hgb','HB','H','h'],
    'Platelets': ['Platelets','Platelets Count','Platelet Count','PLATELET COUNT', 'PLT','Plt','plt','p','P','PC'],
    'White Blood Cells': ['White Blood Cells', 'WBC','Leukocytes','Total Leucocyte Count','wbc'],
    'Red Blood Cells': ['Red Blood Cells','RBC','RBC Count','Erythrocytes'],
    'Hematocrit': ['Hematocrit','Hct','HCT'],
    'Mean Corpuscular Volume': ['Mean Corpuscular Volume','MCV','mcv'],
    'Mean Corpuscular Hemoglobin': ['Mean Corpuscular Hemoglobin', 'MCH','mch'],
    'Mean Corpuscular Hemoglobin Concentration': ['Mean Corpuscular Hemoglobin Concentration','MCHC'],
    'Insulin': ['Insulin','I'],
    'BMI': ['BMI','Body Mass Index'],
    'Systolic Blood Pressure': ['Systolic Blood Pressure','Systolic BP'],
    'Diastolic Blood Pressure': ['Diastolic Blood Pressure','Diastolic BP'],
    'Triglycerides': ['Triglycerides','Trig'],
    'HbA1c': ['HbA1c', 'Glycated Hemoglobin'],
    'LDL Cholesterol': ['LDL Cholesterol','LDL-C'],
    'HDL Cholesterol': ['HDL Cholesterol','HDL-C'],
    'ALT': ['ALT','Alanine Aminotransferase'],
    'AST': ['AST','Aspartate Aminotransferase'],
    'Heart Rate': ['Heart Rate','HR','Pulse'],
    'Creatinine': ['Creatinine','Crea'],
    'Troponin': ['Troponin'],
    'C-reactive Protein': ['C-reactive Protein','CRP']
}
"""

response = model.generate_content([text_prompt, img])
extracted_value = response.text
print("Response received!")

try:
    json_string = extracted_value.strip()
    if json_string.startswith("```json"):
        json_string = json_string[7:]
    if json_string.endswith("```"):
        json_string = json_string[:-3]
    json_string = json_string.strip()
    extracted_values = json.loads(json_string)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from response.text: {e}")
    extracted_values = {}
except AttributeError:
    print("Error: 'response' object not found. Please ensure the previous cell has been run.")
    extracted_values = {}

reference_ranges = {
    "Glucose": {"min": 60, "max": 140},
    "Cholesterol": {"min": 100, "max": 200},
    "Hemoglobin": {"min": 11.5, "max": 17.5},
    "Platelets": {"min": 150000, "max": 400000},
    "White Blood Cells": {"min": 4000, "max": 11000},
    "Red Blood Cells": {"min": 3500000, "max": 5900000},
    "Hematocrit": {"min": 36, "max": 53},
    "Mean Corpuscular Volume": {"min": 80, "max": 100},
    "Mean Corpuscular Hemoglobin": {"min": 25.4, "max": 34.6},
    "Mean Corpuscular Hemoglobin Concentration": {"min": 31, "max": 36},
    "Insulin": {"min": 2, "max": 20},
    "BMI": {"min": 18.5, "max": 24.9},
    "Systolic Blood Pressure": {"min": 90, "max": 120},
    "Diastolic Blood Pressure": {"min": 60, "max": 80},
    "Triglycerides": {"min": 50, "max": 150},
    "HbA1c": {"min": 4.0, "max": 6.0},
    "LDL Cholesterol": {"min": 53, "max": 130},
    "HDL Cholesterol": {"min": 40, "max": 90},
    "ALT": {"min": 7, "max": 56},
    "AST": {"min": 8, "max": 48},
    "Heart Rate": {"min": 60, "max": 100},
    "Creatinine": {"min": 0.6, "max": 1.3},
    "Troponin": {"min": 0, "max": 0.04},
    "C-reactive Protein": {"min": 0, "max": 1.0},
}

normalized_report = {}

for metric, value in extracted_values.items():
    if metric in reference_ranges:
        min_val = reference_ranges[metric]["min"]
        max_val = reference_ranges[metric]["max"]

        if max_val - min_val > 0:
            normalized_value = (value - min_val) / (max_val - min_val)
            normalized_report[metric] = normalized_value
        else:
            normalized_report[metric] = 0.0
    else:
        normalized_report[metric] = "Range Not Found"

df_normalized = pd.DataFrame([normalized_report])

print("Normalized Report Data (0-1 Scale):")
print(df_normalized.to_string())
<<<<<<< HEAD

# new----------------------------
sample_report_data = {
    'Glucose': [np.nan],
    'Cholesterol': [np.nan],
    'Hemoglobin': [np.nan],
    'Platelets': [np.nan],
    'White Blood Cells': [np.nan],
    'Red Blood Cells': [np.nan],
    'Hematocrit': [np.nan],
    'Mean Corpuscular Volume': [np.nan],
    'Mean Corpuscular Hemoglobin': [np.nan],
    'Mean Corpuscular Hemoglobin Concentration': [np.nan],
    'Insulin': [np.nan],
    'BMI': [np.nan],
    'Systolic Blood Pressure': [np.nan],
    'Diastolic Blood Pressure': [np.nan],
    'Triglycerides': [np.nan],
    'HbA1c': [np.nan],
    'LDL Cholesterol': [np.nan],
    'HDL Cholesterol': [np.nan],
    'ALT': [np.nan],
    'AST': [np.nan],
    'Heart Rate': [np.nan],
    'Creatinine': [np.nan],
    'Troponin': [np.nan],
    'C-reactive Protein': [np.nan]
}

# The problem is here: you are trying to use the result of this loop, which is a dict,
# as a DataFrame and then call .fillna() on it.

for key in sample_report_data:
    if key in df_normalized:
        sample_report_data[key] = df_normalized[key].values
    else:
        sample_report_data[key] = [np.nan]

# CORRECT WAY: Convert to DataFrame before imputation
sample_report_df = pd.DataFrame(sample_report_data)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the .pkl file
file_path = os.path.join(script_dir, 'x_train.pkl')

# Load the file
X_train = joblib.load(file_path)
# Step 1: Fill NaNs with column-wise mean
sample_report_imputed = sample_report_df.fillna(X_train.mean())

# Step 2: Replace values outside the range (0, 1) with column-wise mean
for col in sample_report_imputed.columns:
    mean_val = X_train[col].mean()
    mask = (sample_report_imputed[col] < 0) | (sample_report_imputed[col] > 1)
    sample_report_imputed.loc[mask, col] = mean_val

print("\nSample report after imputation:")
print(sample_report_imputed)

# Load the trained model and label encoder
file_path2 = os.path.join(script_dir, 'xgb_disease_model.pkl')
loaded_xgb_model = joblib.load(file_path2)
file_path3 = os.path.join(script_dir, 'label_encoder.pkl')
loaded_label_encoder = joblib.load(file_path3)

# Make predictions on the imputed sample report using the loaded model
predicted_probabilities_imputed = loaded_xgb_model.predict_proba(sample_report_imputed)

# Get the class labels from the loaded label encoder
class_labels = loaded_label_encoder.classes_

# Create a DataFrame to display the probabilities
probabilities_df_imputed = pd.DataFrame(predicted_probabilities_imputed, columns=class_labels)

print("\nPredicted probabilities for each disease (after imputation) using loaded model:")
print(probabilities_df_imputed)
=======
>>>>>>> 49e07b6aa74aa4e7dcdaf5075785ed7392ce61b5
