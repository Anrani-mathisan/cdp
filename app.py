from flask import Flask, render_template, request
import pickle
import numpy as np
import random

app = Flask(__name__)

# Load trained model and preprocessing objects
with open("random_forest.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Job and Poutcome categories (for dropdowns)
job_options = ["Admin", "Technician", "Services", "Management", "Retired",
               "Blue-Collar", "Self-Employed", "Entrepreneur", "Unemployed",
               "Housemaid", "Student"]

poutcome_options = ["Success", "Failure", "Other", "Unknown"]

# Track predictions to enforce 60% "Deposit" and 40% "No Deposit"
prediction_history = {"deposit": 0, "no_deposit": 0}

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            # Extract input data from form
            job = request.form.get("job", "Unknown")  
            balance = float(request.form.get("balance", "0"))
            duration = float(request.form.get("duration", "0"))
            campaign = float(request.form.get("campaign", "0"))
            pdays = float(request.form.get("pdays", "0"))
            previous = float(request.form.get("previous", "0"))
            poutcome = request.form.get("poutcome", "Unknown")  

            # Encode categorical variables
            job_encoded = label_encoders["job"].transform([job])[0] if job in label_encoders["job"].classes_ else 0
            poutcome_encoded = label_encoders["poutcome"].transform([poutcome])[0] if poutcome in label_encoders["poutcome"].classes_ else 0

            # Scale numerical features
            numerical_features = np.array([[duration, pdays, previous, balance, campaign]])
            scaled_features = scaler.transform(numerical_features)

            # Combine numerical & categorical features
            final_features = np.hstack([scaled_features, [[job_encoded, poutcome_encoded]]])

            # Get probability of deposit
            prob = model.predict_proba(final_features)[0][1]

            # Enforce 60% "Deposit" & 40% "No Deposit"
            deposit_count = prediction_history["deposit"]
            no_deposit_count = prediction_history["no_deposit"]
            total_count = deposit_count + no_deposit_count

            if total_count == 0:
                threshold = 0.50  # Start with a 50-50 split
            else:
                deposit_ratio = deposit_count / total_count

                if deposit_ratio < 0.60:
                    threshold = 0.35  # Make deposits more likely
                else:
                    threshold = 0.65  # Make no deposits more likely

            # Make the prediction (add small randomness for variation)
            random_adjustment = random.uniform(-0.05, 0.05)
            final_threshold = threshold + random_adjustment

            prediction = 1 if prob > final_threshold else 0

            # Update prediction history
            if prediction == 1:
                prediction_history["deposit"] += 1
                result = "✅ Customer Will Make a Deposit"
            else:
                prediction_history["no_deposit"] += 1
                result = "❌ No Deposit"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result, job_options=job_options, poutcome_options=poutcome_options)

if __name__ == '__main__':
    app.run(debug=True)
