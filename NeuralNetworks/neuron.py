import random
import ActivationFunction as AF

class neuron:

    def __init__(self,*args):
        self.af_name = "ReLU"
        self.inputs = list()
        self.inputs.append(1)
        self.weights = list()
        self.rawOutput = 0  # the value which didnt enter activationFunction (pure output)
        self.output = 0  # this value entered activation function

        # args[0] = activation_function
        if len(args) == 1 and isinstance(args[0],str):
            self.af_obj=AF.ActivationFunction(args[0])                    #we Created af_obj in constructor
            self.af_name= self.af_obj.getActivationFunction()             #also created af_name in constructor
        # args[0] = activation_function    args[1] = weights
        elif len(args) == 2 and isinstance(args[0],str) and isinstance(args[1],list):
            self.af_obj=AF.ActivationFunction(args[0])
            self.af_name= self.af_obj.getActivationFunction()
            self.weights=args[1]

    def runNeuron(self,Debug=False):
        if Debug :print("ActivationF:",self.af_name)
        if Debug :print("İnputs:", self.inputs)
        if Debug :print("weights:", self.weights)
        self.rawOutput = 0
        for idx,i in enumerate(self.inputs):
            self.rawOutput+= (i * self.weights[idx])

        self.output =0
        self.output = self.af_obj.runActivationFunction(self.af_name, self.rawOutput)
        if Debug :print("output:", self.output)

    def setInputs(self,liste):
        self.inputs[1:]=liste
        if len(self.weights)!=len(self.inputs):
            self.addRandomWeights(len(self.inputs)-len(self.weights))


    def setWeights(self, liste):
        self.weights=liste

    def changeWeight(self,index,value): #tek değişim yapıyor
        self.weights[index]=value

    def getOutput(self):
        return self.output

    def getRawOutput(self):
        return self.rawOutput

    def getActivationName(self):
        return self.af_name

    def getWeights(self):
        return self.weights

    def getInputs(self):
        return self.inputs

    def addRandomWeights(self,weights_number):
        for i in range(weights_number):
            self.weights.append(random.random())