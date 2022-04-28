import math
import numpy as np

class ActivationFunction:
    activation_names = ["ReLU","Sigmoid","Tanh","Softmax"]
    layer_activation_names = ["Softmax"]
    activation_function = None

    def __init__(self,activation_name2):
        self.activation_function = None                   #we created activation_function in constructor
        if(activation_name2 in self.activation_names):   #we check if list include activasion names
            self.activation_function = activation_name2
        else:
            print("please enter a valid Activation Function name, default value is selected as ReLU ("+str(activation_name2)+" --> ReLU)")
            self.activation_function = "ReLU"

    def getActivationFunction(self):
        return self.activation_function


    ############################
    def relu(self,value):
        return max(0,value)

    def reluDerivative(self, value):
        if value <= 0:
            return 0
        elif value>0:
            return 1
    ############################
    def sigmoid(self,value):
        return 1/(1+math.e**(-value))

    def sigmoidDerivative(self, value):
        return self.sigmoid(value)*(1-self.sigmoid(value))

    ############################
    def tanh(self,value):
        return (math.e**(value)-math.e**(-value))/(math.e**(value)+math.e**(-value))
    def tanhDerivative(value):
        return 4/((math.e**(value)+math.e**(-value))**2)

    ############################
    def softmax(self,list):
        e_x = np.exp(list - np.max(list))
        return e_x / e_x.sum(axis=0)  # only difference

    def softmaxDerivative(self, list):
        Sz = self.softmax(list)
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D.diagonal().tolist()    #return D


    def runActivationFunction(self, name, value):
        if (name == self.activation_names[0]):
            return self.relu(value)
        elif (name == self.activation_names[1]):
            return self.sigmoid(value)
        elif (name == self.activation_names[2]):
            return self.tanh(value)

    @staticmethod
    def runActivationFunctionLayer(name,values):
        if (name == ActivationFunction.layer_activation_names[0]):
            return ActivationFunction.softmax(ActivationFunction,values)
        else:
            return values


    def runActivationFunctionDerivative(self, name, value):
        if (name == self.activation_names[0]):
            return self.reluDerivative(value)
        elif (name == self.activation_names[1]):
            return self.sigmoidDerivative(value)
        elif (name == self.activation_names[2]):
            return self.tanhDerivative(value)
       #ADD Softmax deritive
