import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd
import gcsfs

file_path = 'gs://assignment3_bucket_sdb/loan_data.csv'

fs = gcsfs.GCSFileSystem()
with fs.open(file_path, 'r') as f:
    loan_data = pd.read_csv(f)

loan_data['person_gender'] = LabelEncoder().fit_transform(loan_data['person_gender'])
loan_data['person_education'] = LabelEncoder().fit_transform(loan_data['person_education'])
loan_data['person_home_ownership'] = LabelEncoder().fit_transform(loan_data['person_home_ownership'])
loan_data['loan_intent'] = LabelEncoder().fit_transform(loan_data['loan_intent'])
loan_data['previous_loan_defaults_on_file'] = loan_data['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

X = loan_data.drop('loan_status', axis=1)
y = loan_data['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model(X_train.shape[1])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

model_save_path = 'gs://assignment3_bucket_sdb/saved_model.keras'
model.save(model_save_path, save_format='keras')  
