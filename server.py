import sys
sys.path.append("src")

from flask import Flask, render_template
from turbo_flask import Turbo
import threading

from src.network import NeuralNetwork
import numpy as np

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)

nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 16, "activation": "relu"},
    {"input_dim": 16, "output_dim": 1, "activation": "sigmoid"},
]

NN = NeuralNetwork(nn_architecture)

app = Flask(__name__)
turbo = Turbo(app)

@app.route('/')
def index():
    print("Hello")
    return render_template('index.html')

@app.route('/testing/')
def testing():
    
    try:
        c, a, i = NN.step(X.T, y.reshape(-1,1).T, 0.01)
        return render_template('index.html', name=c)
        
    except Exception as er:
        return str(er)

@app.route('/my-link/')
def my_link():
    print ('I got clicked!')
    return 'Click.'

@app.context_processor
def inject_load():
    if sys.platform.startswith('linux'): 
        with open('/proc/loadavg', 'rt') as f:
            load = f.read().split()[0:3]
    else:
        load = [int(np.random.random() * 100) / 100 for _ in range(3)]
    
    c, a, i = NN.step(X.T, y.reshape(-1,1).T, 0.01)
    
    c = f"{c:.7f}"
    a = f"{a:.7f}"
    
    return {'load1': load[0], 'load5': load[1], 'load15': load[2], 'loss':c, 'acc':a, 'epoch':i}

@app.before_first_request
def before_first_request():
    threading.Thread(target=update_load).start()


def update_load():
    with app.app_context():
        while True:
            turbo.push(turbo.replace(render_template('loadavg.html'), 'load'))

if __name__ == '__main__':
  app.run(debug=True)
