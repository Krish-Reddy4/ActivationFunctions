import numpy as np




class Activation:

    def __init__(self):
        pass

    def sigmoid(self, x):
        return (1) / (1 + np.exp(x))

    def relu(self, x):
        return max(0,x)
    
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def leakyrelu(self, x, a):
        return max(a*x, x)

    def softmax(self, l : list):
        es = 0
        softmax_prob = []

        for i in l:
            es += np.exp(i)
        
        for i in  l:
            softmax_prob.append(np.exp(i) / es )
        
        return softmax_prob



