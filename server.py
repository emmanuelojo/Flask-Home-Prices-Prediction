''' 
This code takes the JSON data while POST request performs the prediction  using loaded model and returns 
the results in JSON format
'''

import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    prediction = model.predict([[np.array(data['exp'])]])
    
    output = prediction[0]
    
    return jsonify(output)


if __name__ =='__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin")









