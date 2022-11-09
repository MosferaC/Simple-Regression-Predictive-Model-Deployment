import numpy as np
#from waitress import serve
from flask import Flask, request, jsonify, render_template
import pickle  # we need to read the pickle file

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/') #calling html file to load the pkl file
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST']) #model will take some inputs and would give me some outputs
def predict():   #web api
    
    '''
    For rendering HTML GUI
    
    '''
    
    int_features = [int(x) for x in request.form.values()]  # will take all the values in request form
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text = 'Employee Salary should be in $ {}'.format(output)) # will render the index.html again for replacing the prediction 

if __name__ == "__main__":
    app.run(debug=True)
