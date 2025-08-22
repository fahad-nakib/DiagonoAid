import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



script_dir = os.path.dirname(os.path.abspath(__file__))
X_train = joblib.load(os.path.join(script_dir, 'x_train.pkl'))
dataset = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')
dataset.head()


# For demonstration purposes, let's create a copy of the dataset
# and introduce some missing values artificially
dataset_with_missing = dataset.copy()

# Introduce some random missing values (e.g., in Glucose and Cholesterol columns)
np.random.seed(42) # for reproducibility
missing_indices = np.random.choice(dataset_with_missing.index, size=100, replace=False)
dataset_with_missing.loc[missing_indices, 'Glucose'] = np.nan

missing_indices = np.random.choice(dataset_with_missing.index, size=80, replace=False)
dataset_with_missing.loc[missing_indices, 'Cholesterol'] = np.nan

print("Dataset with artificial missing values:")
print(dataset_with_missing.isnull().sum())

# Impute missing values with the mean of each *numeric* column
numeric_cols = dataset_with_missing.select_dtypes(include=np.number).columns
dataset_imputed = dataset_with_missing.copy()
dataset_imputed[numeric_cols] = dataset_with_missing[numeric_cols].fillna(dataset_with_missing[numeric_cols].mean())


print("\nDataset after imputation with mean:")
print(dataset_imputed.isnull().sum())


from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = dataset_imputed.drop('Disease', axis=1)
y = dataset_imputed['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save X_train
joblib.dump(X_train, 'X_train.pkl')

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)




import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Encode the target variable 'Disease'
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize and train the XGBoost classifier
# Use the encoded labels for training
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42)
xgb_model.fit(X_train, y_train_encoded)

# Make predictions on the test set
y_pred_encoded = xgb_model.predict(X_test)

# Decode the predicted labels back to original disease names
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test_encoded, y_pred_encoded) # Evaluate using encoded labels
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
# Pass original y_test and decoded y_pred to classification_report
print(classification_report(y_test, y_pred))



# Create a sample DataFrame for a new blood test report
# Replace these values with actual data from a user's report
sample_report_data = {
    'Glucose': [0.5],
    'Cholesterol': [0.6],
    'Hemoglobin': [0.7],
    'Platelets': [0.8],
    'White Blood Cells': [0.55],
    'Red Blood Cells': [0.45],
    'Hematocrit': [0.35],
    'Mean Corpuscular Volume': [0.65],
    'Mean Corpuscular Hemoglobin': [0.4],
    'Mean Corpuscular Hemoglobin Concentration': [0.75],
    'Insulin': [0.3],
    'BMI': [0.5],
    'Systolic Blood Pressure': [0.6],
    'Diastolic Blood Pressure': [0.4],
    'Triglycerides': [0.55],
    'HbA1c': [0.6],
    'LDL Cholesterol': [0.5],
    'HDL Cholesterol': [0.65],
    'ALT': [0.3],
    'AST': [0.4],
    'Heart Rate': [0.7],
    'Creatinine': [0.3],
    'Troponin': [0.4],
    'C-reactive Protein': [0.5]
}

sample_report_df = pd.DataFrame(sample_report_data)

# Make predictions (get probability estimates)
# The predict_proba method returns the probability of the sample belonging to each class
predicted_probabilities = xgb_model.predict_proba(sample_report_df)

# Get the class labels
class_labels = label_encoder.classes_

# Create a DataFrame to display the probabilities
probabilities_df = pd.DataFrame(predicted_probabilities, columns=class_labels)

print("Predicted probabilities for each disease:")
print(probabilities_df)




# Create a sample DataFrame for a new blood test report with some missing values
# 1	0.121786	0.023058	0.944893	0.905372	0.507711	0.403033	0.164216	0.307553	0.207938	0.505562	...	0.856810	0.652465	0.106961	0.942549	0.344261	0.666368	0.659060	0.816982	0.401166	Diabetes
sample_report_data_with_missing = {
    'Glucose': [0.121786],  # Introduce a missing Glucose value
    'Cholesterol': [0.023058],
    'Hemoglobin': [0.944893],
    'Platelets': [0.905372],
    'White Blood Cells': [0.507711],
    'Red Blood Cells': [0.403033],
    'Hematocrit': [0.164216],
    'Mean Corpuscular Volume': [0.307553],
    'Mean Corpuscular Hemoglobin': [0.207938],
    'Mean Corpuscular Hemoglobin Concentration': [0.505562],
    'Insulin': [0.571161],
    'BMI': [0.839270],
    'Systolic Blood Pressure': [0.580902],
    'Diastolic Blood Pressure': [0.556037],
    'Triglycerides': [0.477742],
    'HbA1c': [0.856810],  # Introduce a missing HbA1c value
    'LDL Cholesterol': [0.652465],
    'HDL Cholesterol': [0.106961],
    'ALT': [0.942549],
    'AST': [0.344261],
    'Heart Rate': [0.666368],
    'Creatinine': [0.659060],
    'Troponin': [0.816982],
    'C-reactive Protein': [0.401166]
}

sample_report_df_with_missing = pd.DataFrame(sample_report_data_with_missing)

print("Sample report with missing values:")
print(sample_report_df_with_missing)
print("\nMissing values in the sample report:")
print(sample_report_df_with_missing.isnull().sum())

# Impute missing values in the sample report using the mean from the training data
# It's crucial to use the means from the data the model was trained on
sample_report_imputed = sample_report_df_with_missing.fillna(X_train.mean())

print("\nSample report after imputation:")
print(sample_report_imputed)
print("\nMissing values in the sample report after imputation:")
print(sample_report_imputed.isnull().sum())


# Make predictions on the imputed sample report
predicted_probabilities_imputed = xgb_model.predict_proba(sample_report_imputed)

# Get the class labels
class_labels = label_encoder.classes_

# Create a DataFrame to display the probabilities
probabilities_df_imputed = pd.DataFrame(predicted_probabilities_imputed, columns=class_labels)

print("\nPredicted probabilities for each disease (after imputation):")
print(probabilities_df_imputed)

# Save the trained model and label encoder
joblib.dump(xgb_model, 'xgb_disease_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\nTrained XGBoost model and LabelEncoder saved.")