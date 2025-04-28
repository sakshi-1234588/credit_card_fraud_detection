from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/fraud_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        Time = float(request.form['Time'])
        Amount = float(request.form['Amount'])
        V1 = float(request.form['V1'])
        V2 = float(request.form['V2'])

        input_data = [Time, V1, V2] + [0.0]*26 + [Amount]

        final_input = np.array([input_data])

        # Make prediction
        prediction = model.predict(final_input)

        if prediction[0] == 1 or V1 < -15 or V2 >20:
            result = "Alert: Fraudulent Transaction Detected!"
        else:
            result = " Congratulations! Transaction Looks Safe."

        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)