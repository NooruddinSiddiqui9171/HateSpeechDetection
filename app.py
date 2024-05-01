from flask import Flask, render_template, request
from Training.classifier import model
from Training.train_test import cv

app = Flask(__name__, template_folder='template')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    user_input = request.form['user_input']
    output = cv.transform([user_input]).toarray()
    return render_template('result.html', output=model.predict(output))


if __name__ == '__main__':
    app.run(debug=True)
