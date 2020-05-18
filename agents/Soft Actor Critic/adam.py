import numpy as np

class Adam:
    def __init__(self, parameter, alpha=0.0005, b1=0.9, b2=0.999, e=1e-8):
        self.alpha = alpha
        self.beta1 = b1
        self.beta2 = b2
        self.epsilon = e
        self.m = 0
        self.v = 0
        self.t = 0
        self.param = parameter
        
    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)
        m_hat = self.m/(1 - self.beta1**self.t)
        v_hat = self.v/(1 - self.beta2**self.t)
        self.param = self.param - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))
        return self.param