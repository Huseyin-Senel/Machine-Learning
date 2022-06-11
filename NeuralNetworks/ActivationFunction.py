import math
import decimal

import scipy.special as special
from mpmath import mp
import numpy as np

class ActivationFunction:
    activation_names = ["ReLU","Sigmoid","Tanh","Softmax"]


    @staticmethod
    def checkActivationFunction(name):
        if(name in ActivationFunction.activation_names):
            return name
        else:
            print("please enter a valid Activation Function name, default value is selected as ReLU (" + str(name) + " --> ReLU)")
            return "ReLU"
    ############################

    @staticmethod
    def relu(values,index):
        return max(0,values[index])

    @staticmethod
    def reluDerivative(values,index):
        if values[index] <= 0:
            return 0
        elif values[index]>0:
            return 1

    @staticmethod
    def sigmoid(values,index):
        return 1/(1+math.exp(-(values[index]%10)))

    @staticmethod
    def sigmoidDerivative(values,index):
        return special.expit(values)[index]*(1-special.expit(values)[index])


    @staticmethod
    def tanh(values,index):
        #values = values % 10
        #values = ActivationFunction.roundList(values)
        return float((math.exp(values[index]%10)-math.exp(-(values[index]%10)))/(math.exp(values[index]%10)+math.exp(-(values[index]%10)))) #(math.e**(values[index])-math.e**(-values[index]))/(math.e**(values[index])+math.e**(-values[index]))

    @staticmethod
    def tanhDerivative(values,index):
        return 1-ActivationFunction.tanh(values,index)**2


    @staticmethod
    def softmax(values,index):
        for i in values:
            values[index] = math.exp(values[index]%10)
        e_x = values
        return (e_x[index] / e_x.sum())

    @staticmethod
    def softmaxDerivative(values,index):
       return special.softmax(values)[index]*(1-special.softmax(values)[index])

    ############################

    @staticmethod
    def runActivationFunction(name, values,index):
        #values = ActivationFunction.roundList(values)
        if (name == ActivationFunction.activation_names[0]):
            return ActivationFunction.relu(values,index)
        elif (name == ActivationFunction.activation_names[1]):
            #return ActivationFunction.sigmoid(values,index)
            return special.expit(values)[index]
        elif (name == ActivationFunction.activation_names[2]):
            return ActivationFunction.tanh(values,index)
        elif (name == ActivationFunction.activation_names[3]):
            #return ActivationFunction.softmax(values,index)
            return special.softmax(values)[index]


    @staticmethod
    def runActivationFunctionDerivative(name, values,index):
        #values = ActivationFunction.roundList(values)
        if (name == ActivationFunction.activation_names[0]):
            return ActivationFunction.reluDerivative(values,index)
        elif (name == ActivationFunction.activation_names[1]):
            return ActivationFunction.sigmoidDerivative(values,index)
        elif (name == ActivationFunction.activation_names[2]):
            return ActivationFunction.tanhDerivative(values,index)
        elif (name == ActivationFunction.activation_names[3]):
            return ActivationFunction.softmaxDerivative(values,index)


    @staticmethod
    def round(value):
        return float(decimal.Decimal(value).quantize(decimal.Decimal('0.0001')))

    #round 10 digits after the decimal point list
    @staticmethod
    def roundList(list):
        for i in range(len(list)):
            list[i] = ActivationFunction.round(list[i])
        return list