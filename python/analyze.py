import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import base64
import io
from openai import OpenAI
import google.generativeai as genai

# Configure Gemini
# genai.configure(api_key="Place your own api key.....")   # *** IMPORTANT: Use your own API key here ***

# Locate first image in uploads/
uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise FileNotFoundError("No image files found in the uploads folder.")
img_path = os.path.join(uploads_dir, image_files[0])
img = Image.open(img_path)

# Save the image to a bytes buffer (this step is to ensure it's in a format for encoding)
buffer = io.BytesIO()
img.save(buffer, format="PNG") # You might need to specify the original format if it's not PNG
image_bytes = buffer.getvalue()

# 2. Encode the image bytes to a base64 string
encoded_string = base64.b64encode(image_bytes).decode('utf-8')

# Gemini prompt for extracting values
text_prompt = """
Extract numeric values for the following medical terms from the image. Fil missing values with realistic numeric values and convert units like 'lakhs' to numeric form.
medical_terms = {
    'Glucose': ['Glucose','Glu','G','g'],
    'Cholesterol': ['Cholesterol','Chol','C','c'],
    'Hemoglobin': ['Hemoglobin','Haemoglobin','Hb','Hgb','HB','H','h'],
    'Platelets': ['Platelets','Platelets Count','Platelet Count','PLATELET COUNT'],
    'White Blood Cells': ['White Blood Cells','WBC','Leukocytes','Total Leucocyte Count','wbc'],
    'Red Blood Cells': ['Red Blood Cells','RBC','RBC Count','Erythrocytes'],
    'Hematocrit': ['Hematocrit','Hct','HCT'],
    'Mean Corpuscular Volume': ['Mean Corpuscular Volume','MCV','mcv'],
    'Mean Corpuscular Hemoglobin': ['Mean Corpuscular Hemoglobin','MCH','mch'],
    'Mean Corpuscular Hemoglobin Concentration': ['Mean Corpuscular Hemoglobin Concentration','MCHC'],
    'Insulin': ['Insulin','I'],
    'BMI': ['BMI','Body Mass Index'],
    'Systolic Blood Pressure': ['Systolic Blood Pressure','Systolic BP'],
    'Diastolic Blood Pressure': ['Diastolic Blood Pressure','Diastolic BP'],
    'Triglycerides': ['Triglycerides','Trig'],
    'HbA1c': ['HbA1c','Glycated Hemoglobin'],
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

# Extract values using Gemini
# model = genai.GenerativeModel('gemini-2.5-flash')
# response = model.generate_content([text_prompt, img])
# raw_text = response.text.strip()
#print("Raw Gemini response:")
#print(repr(raw_text))
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR OPENROUTER_API_KEY"  # *** IMPORTANT: Use your own API key here ***,
)

completion = client.chat.completions.create(
    extra_body={},
    model="google/gemini-2.5-flash-image-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_string}"
                    }
                }
            ]
        }
    ]
)

raw_text = completion.choices[0].message.content
# Extract JSON block using regex
match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
if match:
    json_block = match.group(1).strip()
    try:
        extracted_values = json.loads(json_block)
    except json.JSONDecodeError as e:
        print("JSON decoding failed:", e)
        extracted_values = {}
else:
    print("No valid JSON block found in Gemini response.")
    extracted_values = {}


# Reference ranges for normalization
reference_ranges = {
    "Glucose": (60, 140), "Cholesterol": (100, 200), "Hemoglobin": (11.5, 17.5),
    "Platelets": (150000, 400000), "White Blood Cells": (4000, 11000),
    "Red Blood Cells": (3500000, 5900000), "Hematocrit": (36, 53),
    "Mean Corpuscular Volume": (80, 100), "Mean Corpuscular Hemoglobin": (25.4, 34.6),
    "Mean Corpuscular Hemoglobin Concentration": (31, 36), "Insulin": (2, 20),
    "BMI": (18.5, 24.9), "Systolic Blood Pressure": (90, 120), "Diastolic Blood Pressure": (60, 80),
    "Triglycerides": (50, 150), "HbA1c": (4.0, 6.0), "LDL Cholesterol": (53, 130),
    "HDL Cholesterol": (40, 90), "ALT": (7, 56), "AST": (8, 48),
    "Heart Rate": (60, 100), "Creatinine": (0.6, 1.3), "Troponin": (0, 0.04),
    "C-reactive Protein": (0, 1.0)
}

# Normalize values
normalized_report = {}
for metric, value in extracted_values.items():
    if metric in reference_ranges and value is not None:
        min_val, max_val = reference_ranges[metric]
        if max_val > min_val:
            normalized_report[metric] = (value - min_val) / (max_val - min_val)
        else:
            normalized_report[metric] = 0.0
    else:
        normalized_report[metric] = np.nan

df_normalized = pd.DataFrame([normalized_report])

# Prepare sample report with NaNs
sample_report_df = pd.DataFrame({key: [df_normalized.get(key, pd.Series([np.nan])).values[0]] for key in reference_ranges})

# Load training data and model
script_dir = os.path.dirname(os.path.abspath(__file__))
X_train = joblib.load(os.path.join(script_dir, 'x_train.pkl'))
xgb_model = joblib.load(os.path.join(script_dir, 'xgb_disease_model.pkl'))
label_encoder = joblib.load(os.path.join(script_dir, 'label_encoder.pkl'))

# Impute missing values and clip out-of-range
sample_report_imputed = sample_report_df.fillna(X_train.mean())
for col in sample_report_imputed.columns:
    mean_val = X_train[col].mean()
    sample_report_imputed[col] = sample_report_imputed[col].clip(0, 1).fillna(mean_val)

# Predict disease probabilities
predicted_probs = xgb_model.predict_proba(sample_report_imputed)
class_labels = label_encoder.classes_
probabilities_df = pd.DataFrame(predicted_probs, columns=class_labels)

# Format for Gemini summary
disease_list = [
    {"name": col, "probability": float(round(probabilities_df[col][0] * 100, 2))}
    for col in probabilities_df.columns
]

summary_prompt = f"""
Based on the following disease probabilities, generate a response in this exact format:

[
  {{
    name: "DiseaseName",
    probability: 35,
    description: "2 to 3 line summary explaining the implication of the probability in short.",
    tips: ["Tip 1", "Tip 2", "Tip 3" shortly in 1 line each"]
  }},
  ...
]

Here is the input data:
{json.dumps(disease_list, indent=2)}
"""


summary_response = client.chat.completions.create(
    extra_body={},
    model="google/gemini-2.5-flash-image-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": summary_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_string}"
                    }
                }
            ]
        }
    ]
)
# summary_response = model.generate_content(summary_prompt)
summary_text = summary_response.choices[0].message.content

# Clean and parse final report
if summary_text.startswith("```json"):
    summary_text = summary_text[7:]
if summary_text.endswith("```"):
    summary_text = summary_text[:-3]
summary_text = summary_text.strip()

try:
    final_report = json.loads(summary_text)
except json.JSONDecodeError as e:
    print("Final report JSON decoding failed:", e)
    final_report = {}

# Final output
#print("This is final report:")
print(json.dumps(final_report, indent=2))
