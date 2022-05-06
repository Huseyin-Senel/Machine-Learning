import math
import numpy as np

class ActivationFunction:
    activation_names = ["ReLU","Sigmoid","Tanh","Softmax"]
    layer_activation_names = ["Softmax"]


    @staticmethod
    def checkActivationFunction(name):
        if(name in ActivationFunction.activation_names):
            type = "AF"
            if (name in ActivationFunction.layer_activation_names):
                type = "LAF"
            return name , type
        else:
            print("please enter a valid Activation Function name, default value is selected as ReLU (" + str(name) + " --> ReLU)")
            return "ReLU","AF"

    ############################

    @staticmethod
    def relu(value):
        return max(0,value)
    @staticmethod
    def reluDerivative(value):
        if value <= 0:
            return 0
        elif value>0:
            return 1


    @staticmethod
    def sigmoid(value):
        return 1/(1+math.e**(-value))
    @staticmethod
    def sigmoidDerivative(value):
        return ActivationFunction.sigmoid(value)*(1-ActivationFunction.sigmoid(value))


    @staticmethod
    def tanh(value):
        return (math.e**(value)-math.e**(-value))/(math.e**(value)+math.e**(-value))
    @staticmethod
    def tanhDerivative(value):
        return 4/((math.e**(value)+math.e**(-value))**2)


    @staticmethod
    def softmax(list):
        e_x = np.exp(list - np.max(list))
        return e_x / e_x.sum(axis=0)  # only difference
    @staticmethod
    def softmaxDerivative(list):
        Sz = ActivationFunction.softmax(list)
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D.diagonal().tolist()    #return D

    ############################

    @staticmethod
    def runActivationFunction(name, value,rawNeuronOutputs=None):
        if (name == ActivationFunction.activation_names[0]):
            return ActivationFunction.relu(value)
        elif (name == ActivationFunction.activation_names[1]):
            return ActivationFunction.sigmoid(value)
        elif (name == ActivationFunction.activation_names[2]):
            return ActivationFunction.tanh(value)
        elif (name == ActivationFunction.activation_names[3]):
            return ActivationFunction.softmax(rawNeuronOutputs)


    @staticmethod
    def runActivationFunctionDerivative(name, value,list=None):
        if (name == ActivationFunction.activation_names[0]):
            return ActivationFunction.reluDerivative(value)
        elif (name == ActivationFunction.activation_names[1]):
            return ActivationFunction.sigmoidDerivative(value)
        elif (name == ActivationFunction.activation_names[2]):
            return ActivationFunction.tanhDerivative(value)
        elif (name == ActivationFunction.activation_names[3]):
            return ActivationFunction.softmaxDerivative(list)
