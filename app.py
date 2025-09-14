from flask import Flask, render_template,request
import numpy as np
import pickle

app = Flask(__name__)

#route

@app.route('/')

def index_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_price():
    Kms = float(request.form['km'])
    Age = float(request.form['age'])
    Op = float(request.form['op'])
    Fuel_Type = request.form['fuel_type']

    if Fuel_Type == 'Petrol':
        fuel = [0.0, 0.0, 1.0]
    elif Fuel_Type == 'Diesel':
        fuel = [0.0, 1.0, 0.0]
    else:
        fuel = [1.0, 0.0, 0.0]

    transmission = request.form['transmission']
    if transmission == 'Manual':
        transmission = [0.0,1.0]
    else:
        transmission = [1.0,0.0]

    data=[np.array([Kms,Op,Age,fuel[0],fuel[1],
                    fuel[2],transmission[0],transmission[1]])]

    model = pickle.load(open('./models/lr_model.pkl','rb'))

    result=np.round(model.predict(data))

    msg = "The price of the car is ₹" + str(result[0])

    return render_template('index.html', prediction=msg)

# run the app
if __name__ == '__main__':   
    app.run(debug=True)
