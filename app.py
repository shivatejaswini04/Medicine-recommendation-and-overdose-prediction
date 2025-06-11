from flask import Flask, render_template, request
import joblib
import pandas as pd
import os


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load model and encoders
model = joblib.load("models/overdose_model.pkl")
le_drug = joblib.load("models/label_encoder_drug.pkl")
le_age = joblib.load("models/label_encoder_age.pkl")

# Load CSV dataset
csv_path = os.path.join("datasets", "drug_overdose.csv")
df = pd.read_csv(csv_path)

safe_limits = {}
available_drugs = sorted(df['Drug Name'].unique())
drug_name_map = {drug.lower(): drug for drug in available_drugs}

for _, row in df.iterrows():
    safe_limits.setdefault(row['Drug Name'], {})[row['Age Group']] = {
        'safe_limit': row['Safe Daily Limit (mg)'],
        'max_frequency': row['Drug Frequency per Day']
    }

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/overdose_prediction', methods=['GET', 'POST'])
def overdose_prediction():
    result = None
    error_message = None

    if request.method == 'POST':
        drug_input = request.form['drug'].strip()
        drug = drug_name_map.get(drug_input.lower())
        age_group = request.form['age_group']
        dosage = request.form.get('dosage')
        frequency = request.form.get('frequency')

        if not drug:
            error_message = f"Invalid drug '{drug_input}'. Available drugs: {', '.join(available_drugs)}"
        else:
            try:
                dosage = float(dosage)
                frequency = int(frequency)

                if dosage <= 0 or frequency <= 0:
                    raise ValueError("Values must be positive.")

                is_overdose = False
                reason = ""

                if drug in safe_limits and age_group in safe_limits[drug]:
                    drug_info = safe_limits[drug][age_group]
                    safe_limit = drug_info['safe_limit']
                    max_freq = drug_info['max_frequency']
                    total = dosage * frequency

                    if dosage > safe_limit:
                        is_overdose = True
                        reason = "Single dose exceeds limit."
                    elif frequency > max_freq:
                        is_overdose = True
                        reason = "Frequency too high."
                    elif total > safe_limit * max_freq:
                        is_overdose = True
                        reason = "Total daily dose too high."

                result = "Overdosed" if is_overdose else "Safe"
                if is_overdose:
                    print("⚠️", reason)

            except ValueError:
                error_message = "Please enter valid numbers for dosage and frequency."

    return render_template('overdose.html',
                           result=result,
                           available_drugs=available_drugs,
                           error_message=error_message,
                           submitted_values=request.form if request.method == 'POST' else None)

if __name__ == '__main__':
    app.run(port=5001)
