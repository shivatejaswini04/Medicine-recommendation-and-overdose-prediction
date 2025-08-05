# Medicine Recommendation and Overdose Prediction

This project is a web-based application for recommending medicines, predicting overdose risks, and providing health-related information using machine learning and a chatbot powered by Cohere.

## Features
- Disease prediction based on symptoms
- Medicine, diet, workout, and precaution recommendations
- User authentication (login/register)
- Chatbot for health queries (Cohere API)
- Disease information pages
- Overdose prediction

## Technologies Used
- Python (Flask, numpy, pandas, pickle)
- MySQL (user authentication, data storage)
- Cohere API (chatbot)
- HTML/CSS (templates/static)
- Machine Learning (SVC model)

## Project Structure
```
app.py, main.py, chatbot.py, train_model.py
models/                # ML models and encoders
static/                # Images and CSS
templates/             # HTML templates for all pages
 datasets/             # CSV files for symptoms, medications, diets, etc.
```

## Setup Instructions
1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up MySQL and create the `health_db` database with a `users` table.
4. Add your Cohere API key to the `.env` file:
   ```
   COHERE_API_KEY=YOUR_API_KEY_HERE
   ```
5. Run the application:
   ```
   python main.py
   ```
6. Access the app at `http://localhost:5000`

## Usage
- Register and log in to access features.
- Enter symptoms to get disease predictions and recommendations.
- Use the chatbot for health-related questions.

## Security
- The `.env` file is ignored in version control for API key safety.

## License
This project is for educational purposes.
