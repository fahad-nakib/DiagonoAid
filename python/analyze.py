from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
import PIL.Image
import os
import google.generativeai as genai
genai.configure(api_key="AIzaSyDvwICRzk-Replace your own gemini api key")

#img=PIL.Image.open('blood2.png')

# Locate the first image in the uploads folder

uploads_dir = 'uploads'
image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise FileNotFoundError("No image files found in the uploads folder.")
first_image_path = os.path.join(uploads_dir, image_files[0])

# Load the image
img = PIL.Image.open(first_image_path)

model = genai.GenerativeModel('gemini-2.5-flash')

# response=model.generate_content(img)
# print(response.text)


text_prompt = """
extract these medical terms actual numeric value without unit(unit lakhs should be in numeric form) from the image if available
and fill the missing value with appropriate value according to age, gender and other condition but not left any value empty .
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

# Pass both the text and the image as a list
response = model.generate_content([text_prompt, img])

print(response.text)
extracted_value = response.text


# 2nd PHASE AND ITS LIBRARIES

# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import numpy as np
# import json


# Parse the JSON string from response.text into a dictionary
try:
    # Extract the JSON string from the response text
    json_string = extracted_value.strip()
    # Remove potential markdown code block delimiters
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
# Define the reference ranges for each metric
# These ranges are based on standard clinical guidelines for healthy adults
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

# Create an empty dictionary to store the normalized values
normalized_report = {}

# Iterate through the report values to normalize each metric
for metric, value in extracted_values.items():
    if metric in reference_ranges:
        min_val = reference_ranges[metric]["min"]
        max_val = reference_ranges[metric]["max"]

        # Check if the range is valid to avoid division by zero
        if max_val - min_val > 0:
            normalized_value = (value - min_val) / (max_val - min_val)
            normalized_report[metric] = normalized_value
        else:
            # Handle cases with a zero-length range, if any
            normalized_report[metric] = 0.0
    else:
        # Handle cases where a metric is not in the reference ranges
        normalized_report[metric] = "Range Not Found"

# Convert the dictionary to a pandas DataFrame for a cleaner output
df_normalized = pd.DataFrame([normalized_report])

# Display the normalized results
print("Normalized Report Data (0-1 Scale):")
print(df_normalized.to_string())
