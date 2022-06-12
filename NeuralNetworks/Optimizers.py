import numpy as np

class AdamOptimizer:

    def __init__(self, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0


    def update(self,gt,wt):
        self.m = self.beta1*self.m + (1-self.beta1)*gt
        self.v = self.beta2*self.v + (1-self.beta2)*(gt**2)
        m_hat = self.m/(1-self.beta1**self.t)
        v_hat = self.v/(1-self.beta2**self.t)
        wt = wt - self.lr*m_hat/(np.sqrt(v_hat)+self.epsilon)
        return wt

    def updateT(self):
        self.t += 1


