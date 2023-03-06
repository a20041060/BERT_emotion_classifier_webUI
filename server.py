from flask import Flask, redirect, url_for, request,render_template
from bert_classifier import classifier, init_model
import requests
app = Flask(__name__)

url = 'http://127.0.0.1:5000'

# POST/form data
payload = {
    'nm': 'hello world',
}

model = init_model()
@app.route('/')
def index():
   init_classifier(url,payload)
   return render_template('index.html')

def init_classifier(url,data):
   requests.post(url, data)

@app.route('/',methods = ['POST'])
def text():
    if request.method == 'POST':
      text = request.form['nm']
      result =classifier(text,model)
      print(result)
      return render_template("index.html",result = result)


if __name__ == '__main__':
   app.run(debug = True)
