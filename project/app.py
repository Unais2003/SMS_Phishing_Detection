from flask import Flask, render_template, request
from model import predict_spam  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']  
    prediction = predict_spam(url)  
    return render_template('index.html', prediction=prediction, url=url)

if __name__ == '__main__':
    app.run(debug=True)
