from flask import Flask,jsonify,request
from flask_cors import CORS
from lvq import LVQ
from neuralnet import Neural
from layer import Layer
import numpy as np
import json
app = Flask(__name__)
CORS(app)

pcp = Neural(4)
L = Layer(81,4,'sigmoid')
pcp.add_layer(L)

lvq = LVQ(n_class=4,input_shape=81,distance='euclidean')
@app.route('/')
def root():
    print("GET")
    return jsonify({'hello': 'world'})

@app.route("/train",methods=["POST"])
def train():
    print("POST")
    content = request.json
    data = np.array(content['data'])
    label  = np.array(content['label'])
    print(data.shape)
    print(label.shape)
    ev_set = (data,label)
    print(data)
    print(label)
    pcp.fit(X=data,y=label,epochs=100,learn_rate=0.001,eval_set=ev_set)
    return jsonify({'hello': 'world'})

@app.route("/predict",methods=['POST'])
def pred():
    print("SEND PREDICT")
    content = request.json
    data = np.array(content['data'])
    print(data)
    print(pcp.predict(data))
    res = pcp.predict(data)
    res = [ int(e) for e in res]
    return jsonify({'res':res})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)