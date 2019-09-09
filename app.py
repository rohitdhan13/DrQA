#!flask/bin/python
from flask import Flask, request, jsonify
from drqa.reader import Predictor

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    return "healthy!"

@app.route('/predict',methods=['GET'])
def process():
    data = request.json
    document = data['document']
    #print(document)
    question = data['question']
    #print(question)
    predictor = Predictor(None,'spacy',num_workers=0,normalize=True)
    predictions = predictor.predict(document,question,None,1)
    val = []
    for i, p in enumerate(predictions, 1):
        val.append(p[0])
    return jsonify(val)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
