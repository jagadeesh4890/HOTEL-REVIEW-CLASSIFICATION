# Hotel Review Classification

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


app = Flask(__name__)
clf = pickle.load(open("Hotel_Rating_MLogReg.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = clf.predict(data)
    return render_template('result.html',prediction = my_prediction)

message = ['this is good']
clf.predict(message)


if __name__ == '__main__':
    app.run(debug=True)


