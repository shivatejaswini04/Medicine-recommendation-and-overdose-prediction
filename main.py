from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pandas as pd
import pickle
import os
import mysql.connector
import cohere
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Cohere API setup
COHERE_API_KEY = "8YgkYnRpWZgV4nCE9n8mjAzTTxMG4ii3rEp9wR1X"
co = cohere.Client(COHERE_API_KEY)
chat_history = []

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="password",  # Replace with your MySQL password
    database="health_db"  # Replace with your actual database name
)
cursor = conn.cursor()

# Load datasets
sym_des = pd.read_csv("datasets/symptoms_df.csv")
precautions_df = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications_df = pd.read_csv("datasets/medications.csv")
diets_df = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Dictionaries
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Helper function
def helper(dis):
    pre = precautions_df[precautions_df['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]
    med = medications_df[medications_df['Disease'] == dis]['Medication'].values.flatten().tolist()
    die = diets_df[diets_df['Disease'] == dis]['Diet'].values.flatten().tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.flatten().tolist()
    return pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route('/')
def home():
    if 'user' in session:
        return render_template("index.html")
    else:
        return redirect(url_for("login"))


@app.route('/disease_info')
def disease_info():
    return render_template('disease_info.html')

@app.route("/malaria.html")
def malaria():
    return render_template("malaria.html")

@app.route('/Hypothyroidism.html')
def Hypothyroidism():
    return render_template('Hypothyroidism.html')

@app.route('/Psoriasis.html')
def Psoriasis():
    return render_template('Psoriasis.html')

@app.route('/gerd.html')
def gerd():
    return render_template('gerd.html')

@app.route('/Chronic Cholestasis.html')
def chronic_cholesterol():
    return render_template('Chronic Cholestasis.html')

@app.route('/hepatitis a.html')
def hepatitis_a():
    return render_template('hepatitis a.html')

@app.route('/Osteoarthristis.html')
def osteoarthritis():
    return render_template('Osteoarthristis.html')

@app.route('/Paroxysmal Vertigo.html')
def paroxysmal_vertigo():
    return render_template('Paroxysmal Vertigo.html')

@app.route('/Hypoglycemia.html')
def hypoglycemia():
    return render_template('Hypoglycemia.html')

@app.route('/precautions')
def precautions_page():
    return render_template('precautions.html')

@app.route('/diets')
def diets_page():
    return render_template('diets.html')

@app.route('/medications')
def medications_page():
    return render_template('medications.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    global chat_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input.strip():
            chat_history.append(("You", user_input))

            if user_input.lower() in ["hi", "hello", "hey", "start", "hi!", "hello!"]:
                response = "Hi! What can I help you with today?"
            else:
                result = co.generate(
                    model='command-xlarge',
                    prompt=user_input,
                    max_tokens=100
                )
                response = result.generations[0].text.strip()

            chat_history.append(("Bot", response))
        else:
            response = "Please ask a valid question."
    return render_template("chatbot.html", chat_history=chat_history)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        # Query the database to check if credentials are valid
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()

        if user:
            session['user'] = username  # Set the session user
            return redirect(url_for("home"))  # Redirect to the home page after login
        else:
            flash("Invalid credentials")

    return render_template("login.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords do not match")
            return render_template("register.html")

        # Insert into the database
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        existing_user = cursor.fetchone()
        if existing_user:
            flash("Username already exists. Please choose a different one.")
        else:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            flash("Registration successful! You can now log in.")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route('/logout')
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')
    if symptoms:
        symptoms_list = symptoms.split(',')  # Assuming symptoms are sent as a comma-separated string
        disease = get_predicted_value(symptoms_list)
        # Fetch precautions, medications, diets, and workouts for the predicted disease
        precautions, medications, diets, workouts = helper(disease)
        return render_template('result.html', disease=disease, precautions=precautions, medications=medications, diets=diets, workouts=workouts)
    else:
        flash("Please enter symptoms")
        return redirect(url_for("home"))

if __name__ == '__main__':
    app.run(port=5000)
