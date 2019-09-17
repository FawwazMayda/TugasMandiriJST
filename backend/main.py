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

mlp = Neural(4)
mlp.add_layer(Layer(81,20,'relu'))
mlp.add_layer(Layer(20,10,'relu'))
mlp.add_layer(Layer(10,4,'sigmoid'))
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
    ev_set = (data,label)
    pcp.fit(X=data,y=label,epochs=400,learn_rate=0.001,eval_set=ev_set)
    mlp.fit(X=data,y=label,epochs=400,learn_rate=0.001,eval_set=ev_set)
    lvq.fit(data,label,epochs=100,lr=0.0001,eval_set=ev_set)
    return jsonify({'hello': 'world'})

@app.route("/predict",methods=['POST'])
def pred():
    print("SEND PREDICT")
    content = request.json
    data = np.array(content['data'])
    res_pcp = [ int(e) for e in pcp.predict(data)]
    res_mlp = [ int(e) for e in mlp.predict(data)]
    res_lvq = [ int(e) for e in lvq.predict(data)]
    print(res_lvq)
    print(res_mlp)
    print(res_pcp)
    return jsonify({'status':'OK','lvq':res_lvq,'perceptron':res_pcp,'mlp':res_mlp})

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)