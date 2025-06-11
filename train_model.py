import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
csv_path = "datasets/drug_overdose.csv"
df = pd.read_csv(csv_path)

# Simulate dosage per intake (random realistic values)
np.random.seed(42)
df['Dosage per Intake (mg)'] = np.random.randint(50, 1000, size=len(df))

# Calculate total daily intake
df['Total Daily Intake (mg)'] = df['Dosage per Intake (mg)'] * df['Drug Frequency per Day']

# Create 'Overdosed' label
def is_overdose(row):
    if row['Total Daily Intake (mg)'] > row['Safe Daily Limit (mg)']:
        return 1
    if row['Drug Frequency per Day'] > 4:
        return 1
    if row['Dosage per Intake (mg)'] > (0.8 * row['Safe Daily Limit (mg)']):
        return 1
    return 0

df['Overdosed'] = df.apply(is_overdose, axis=1)

# Add extreme cases
extreme_cases = []
for _, row in df.iterrows():
    if row['Drug Name'] == 'Paracetamol':
        extreme_cases.append({
            'Drug Name': row['Drug Name'],
            'Age Group': row['Age Group'],
            'Safe Daily Limit (mg)': row['Safe Daily Limit (mg)'],
            'Drug Frequency per Day': 50,
            'Dosage per Intake (mg)': 600,
            'Total Daily Intake (mg)': 600 * 50,
            'Overdosed': 1
        })
        extreme_cases.append({
            'Drug Name': row['Drug Name'],
            'Age Group': row['Age Group'],
            'Safe Daily Limit (mg)': row['Safe Daily Limit (mg)'],
            'Drug Frequency per Day': 2,
            'Dosage per Intake (mg)': row['Safe Daily Limit (mg)'],
            'Total Daily Intake (mg)': row['Safe Daily Limit (mg)'] * 2,
            'Overdosed': 1
        })

df = pd.concat([df, pd.DataFrame(extreme_cases)], ignore_index=True)

# Encode categorical variables
le_drug = LabelEncoder()
le_age = LabelEncoder()
df['Drug Name Encoded'] = le_drug.fit_transform(df['Drug Name'])
df['Age Group Encoded'] = le_age.fit_transform(df['Age Group'])

# Features and labels
features = ['Drug Name Encoded', 'Age Group Encoded', 'Dosage per Intake (mg)', 'Drug Frequency per Day']
X = df[features]
y = df['Overdosed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/overdose_model.pkl")
joblib.dump(le_drug, "models/label_encoder_drug.pkl")
joblib.dump(le_age, "models/label_encoder_age.pkl")

print("✅ Model and encoders saved in /models/")
