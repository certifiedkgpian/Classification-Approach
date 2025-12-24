from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model safely
MODEL_PATH = 'zinc_model.joblib'
if os.path.exists(MODEL_PATH):
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
else:
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    
    if request.method == 'POST':
        if not model:
            return render_template('index.html', prediction="Error: Model not loaded.")
        
        try:
            # Get values from HTML form
            input_data = pd.DataFrame([[
                float(request.form['l1']),
                float(request.form['l2']),
                float(request.form['thickness']),
                float(request.form['jig_wt']),
                float(request.form['coating'])
            ]], columns=features)
            
            # Predict
            scaled_data = scaler.transform(input_data)
            result = model.predict(scaled_data)[0]
            prediction_text = f"{result}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)