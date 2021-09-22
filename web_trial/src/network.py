import logging as log
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import utils as utils

class DataLoader():
    def __init__(self) -> None:
        pass
    
    

class NeuralNetwork():
    def __init__(self, network_architecture) -> None:
        self.network = network_architecture
        self.init_layer()
        self.epoch = 0

    def init_layer(self, seed=99):
        np.random.seed(seed)
        params_ = {}        

        for idx, layer in enumerate(self.network):
            layer_index = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            
            params_[f"W{str(layer_index)}"] = np.random.randn(layer_output_size, layer_input_size) * 0.1
            params_[f"b{str(layer_index)}"] = np.random.randn(layer_output_size, 1) * 0.1
            
        self.params = params_

    def single_forward(self, A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
    
        if activation is "relu":
            activation_func = utils.relu
        elif activation is "sigmoid":
            activation_func = utils.sigmoid
        else:
            raise Exception('Non-supported activation function')
            
        return activation_func(Z_curr), Z_curr

    def full_forward(self, X):
        memory = {}
        A_curr = X
        
        self.a = A_curr[0][0]
        
        for idx, layer in enumerate(self.network):
            layer_idx = idx + 1
            A_prev = A_curr
            
            act_function = layer["activation"]
            W_curr = self.params[f"W{str(layer_idx)}"]
            b_curr = self.params[f"b{str(layer_idx)}"]
    
            A_curr, Z_curr = self.single_forward(A_prev, W_curr, b_curr, act_function)

            memory[f"A{str(idx)}"] = A_prev
            memory[f"Z{str(layer_idx)}"] = Z_curr

        self.memory = memory
        return A_curr, memory
        
    def single_backard(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        m = A_prev.shape[1]
        
        if activation is "relu":
            backward_activation_func = utils.relu_backward
        elif activation is "sigmoid":
            backward_activation_func = utils.sigmoid_backward
        else:
            raise Exception('Non-supported activation function')
        
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward(self, Y_hat, Y, memory, params_values, nn_architecture):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
        
        dA_prev = utils.BCE_backward(Y_hat, Y)
        
        for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
            
            dA_curr = dA_prev
            
            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]
            
            dA_prev, dW_curr, db_curr = self.single_backard(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
            
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

            self.gradients = grads_values
        
        return grads_values

    def update(self, learning_rate):
        # iteration over network layers
        for layer_idx, layer in enumerate(self.network, 1):
            self.params["W" + str(layer_idx)] -= learning_rate * self.gradients["dW" + str(layer_idx)]        
            self.params["b" + str(layer_idx)] -= learning_rate * self.gradients["db" + str(layer_idx)]

    def step(self, X, Y, learning_rate):
        self.epoch += 1
        
        i = self.epoch
        
        Y_hat, cashe = self.full_forward(X)
        
        cost = utils.BCE(Y_hat, Y)
        accuracy = utils.get_accuracy_value(Y_hat, Y)
        
        grads_values = self.full_backward(Y_hat, Y, cashe, self.params, self.network)
        
        self.update(learning_rate)
        
        return cost, accuracy, i



if __name__ == "__main__":
    nn_architecture = [
        {"input_dim": 2, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 128, "activation": "relu"},
        {"input_dim": 128, "output_dim": 64, "activation": "relu"},
        {"input_dim": 64, "output_dim": 16, "activation": "relu"},
        {"input_dim": 16, "output_dim": 1, "activation": "sigmoid"},
    ]
    
    N_SAMPLES = 1000

    TEST_SIZE = 0.1

    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
    
    logger.add("aaa.log", level="INFO", format="{time} {name} {message}")
    
    
    NN = NeuralNetwork(nn_architecture)
    
    for _ in range(100):
        c, a, it = NN.step(X.T, y.reshape(-1,1).T, 0.01)
        logger.info(f"Loss: {c:.5f}")