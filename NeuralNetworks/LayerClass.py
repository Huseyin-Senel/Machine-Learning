
import numpy as np
import neuron as NU
import ActivationFunction as AF

class Layer:


    def __init__(self,*args):

        self.neurons = list() # K neuron
        self.inputs = list()  # N input
        self.outputs = list()  # M output
        self.weights = np.array([[],[]])  # N x K matrix
        self.biases = list()  # M
        self.newWeights = np.array([[],[]])  # N x K matrix
        self.newBiases = list()  # M

        # args[0] =Neuron Count
        if len(args) == 1 and isinstance(args[0], int):
            for i in range(args[0]):
                self.neurons.append(NU.neuron("ReLU"))
        # args[0]=Neuron Count  ,  args[1] = activation Function
        elif len(args) == 2 and isinstance(args[1], str):
            for i in range(args[0]):
                self.neurons.append(NU.neuron(args[1]))
        # args[0]=Neuron Count  ,  args[1] = activation Function list
        elif len(args) == 2 and isinstance(args[1], list):
            for i in range(args[0]):
                self.neurons.append(NU.neuron(args[1][i]))

    # a = np.matmul([[1,2],[3,4],[5,6],[7,8]],(np.array([1,2]).T))+[1,2,3,4]
    # print("fast calc:",a)
    def runLayer(self,Debug=False):
        if not (len(self.inputs) == self.weights.shape[1] or (len(self.neurons) == self.weights.shape[0])):
            if(Debug):print("\nError: Inputs and weights are not the same size")
            if(Debug):print("Generating random weights...")
            self.createRandomWeights(len(self.inputs))
            if(Debug):print("Weights generated:"+str(self.weights.shape))
        if not (len(self.neurons) == len(self.biases)):
            if(Debug):print("\nError: Neuron count and bias count are not the same size")
            if(Debug):print("Generating random bias...")
            self.createRandomBias()
            if(Debug):print("Biases generated:"+str(len(self.biases)))

        RawOutputs = np.matmul(self.weights,(np.array(self.inputs).T))+self.biases
        self.outputs.clear()
        for i,neuron in enumerate(self.neurons):
            if (Debug):
                print("------Running Neuron"+str(i+1)+"------")
                print("Inputs:",self.inputs)
                print("Weights:",self.weights[i])
                print("Bias:", self.biases[i])
            neuron.setRawOutput(RawOutputs[i])
            if neuron.getActivationType() == "AF":
                neuron.setOutput(AF.ActivationFunction.runActivationFunction(neuron.getActivationName(),RawOutputs[i]))
            elif neuron.getActivationType() == "LAF":
                neuron.setOutput(AF.ActivationFunction.runActivationFunction(neuron.getActivationName(), RawOutputs[i],RawOutputs)[i])

            if (Debug):
                print("Raw Output:",neuron.getRawOutput())
                print("Output:",neuron.getOutput())
            self.outputs.append(self.neurons[i].getOutput())


    def setWeights(self,weights):
        self.weights = weights


    def setInputs(self,liste): #multiple change
        self.inputs=liste

    def getInputs(self):
        return self.inputs

    # args[0] =Neuron, args[1] = insert index
    def insertNeuron(self, *args):
        if len(args) == 1 and isinstance(args[0], NU.neuron):
            self.neurons.append(args[0])
        elif len(args) == 2 and isinstance(args[0], NU.neuron) and isinstance(args[1], int):
            self.neurons.insert(args[1], args[0])

    def deleteNeuron(self,*args):
        if len(args) == 0:
            del self.neurons[-1]
        elif len(args) == 1 and isinstance(args[0], int):
            self.neurons.pop(args[0])

    def changeNeuron(self,index,neuron):
        self.neurons[index] = neuron

    def getNeurons(self):
        return self.neurons

    def getOutputs(self):
        return self.outputs

    def getActivationNames(self):
        names=list()
        for i in range(len(self.neurons)):
            names.append(self.neurons[i].getActivationName())
        return names

    def getNeuronCount(self):
        return len(self.neurons)

    def getNeuron(self,index):
        return self.neurons[index]

    def createRandomWeights(self,inputCount):
        self.weights = np.random.rand(len(self.neurons), inputCount)

    def createRandomBias(self):
        self.biases = np.random.rand(len(self.neurons))

    def getWeights(self):
        return self.weights

    def changeWeight(self,indexs,weight):
        self.weights[indexs[0]][indexs[1]] = weight

    def getBiases(self):
        return self.biases

    def setBiases(self,biases):
        self.biases = biases

    def getNewWeights(self):
        return self.newWeights

    def setNewWeights(self,newWeights):
        self.newWeights = newWeights

    def changeNewWeight(self,indexs,weight):
        self.newWeights[indexs[0]][indexs[1]] = weight

    def getNewBiases(self):
        return self.newBiases

    def setNewBiases(self,newBiases):
        self.newBiases = newBiases

    def changeNewBias(self,index,bias):
        self.newBiases[index] = bias

