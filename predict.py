import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gcsfs

file_path = 'gs://assignment3_bucket_sdb/loan_data.csv'  

loan_data = pd.read_csv(file_path)

def preprocess_input(data):
    data['person_gender'] = LabelEncoder().fit_transform(data['person_gender'])
    data['person_education'] = LabelEncoder().fit_transform(data['person_education'])
    data['person_home_ownership'] = LabelEncoder().fit_transform(data['person_home_ownership'])
    data['loan_intent'] = LabelEncoder().fit_transform(data['loan_intent'])
    data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})
    
    if 'loan_status' in data.columns:
        data = data.drop('loan_status', axis=1)
    
    return data

loan_data_processed = preprocess_input(loan_data)
numeric_columns = loan_data_processed.select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
scaler.fit(loan_data_processed[numeric_columns]) 

new_data = pd.DataFrame({
    'person_age': [30],
    'person_gender': ['female'],
    'person_education': ['Bachelor'],
    'person_income': [55000],
    'person_emp_exp': [5],
    'person_home_ownership': ['RENT'],
    'loan_amnt': [20000],
    'loan_intent': ['PERSONAL'],
    'loan_int_rate': [12.5],
    'loan_percent_income': [0.36],
    'cb_person_cred_hist_length': [3],
    'credit_score': [680],
    'previous_loan_defaults_on_file': ['No']
})

new_data_processed = preprocess_input(new_data)
new_data_processed[numeric_columns] = scaler.transform(new_data_processed[numeric_columns])

model_path = 'gs://assignment3_bucket_sdb/saved_model.keras'  
with fs.open(model_path, 'rb') as f:
    model = tf.keras.models.load_model(f)

predictions = model.predict(new_data_processed)
predicted_class = (predictions > 0.5).astype("int32") 

print(f"Predicted class: {predicted_class[0][0]}")





