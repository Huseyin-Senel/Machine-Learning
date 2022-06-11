
import numpy as np
import neuron as NU
import ActivationFunction as AF

class Layer:


    def __init__(self,*args):

        self.neurons = list() # K neuron
        self.inputs = list()  # N input
        self.rawOutputs = list() # K
        self.outputs = list()  # K
        self.weights = np.array([[],[]])  # K x N matrix  row x column
        self.newWeights = np.array([[], []])  # K x N matrix
        self.deltaWeights = np.array([[], []])  # K x N matrix
        self.biases = list()  # K
        self.newBiases = list()  # K
        self.deltaBiases = list()  # K

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

    def runLayer(self,Debug=False):
        if not(len(self.inputs) == self.weights.shape[1]) or not(len(self.neurons) == self.weights.shape[0]):
            if(Debug):print("\nError: Inputs and weights are not the same size")
            if(Debug):print("Generating random weights...")
            self.createRandomWeights()
            if(Debug):print("Weights generated:"+str(self.weights.shape))
        if not (len(self.neurons) == len(self.biases)):
            if(Debug):print("\nError: Neuron count and bias count are not the same size")
            if(Debug):print("Generating random bias...")
            self.createRandomBias()
            if(Debug):print("Biases generated:"+str(len(self.biases)))
        if not (len(self.inputs)==self.deltaWeights.shape[1]) or not (len(self.neurons)==self.deltaWeights.shape[0]):
            self.deltaWeights = np.zeros((len(self.neurons),len(self.inputs)))
            print("bbbb")
        if not (len(self.inputs)==len(self.deltaBiases)):
            self.deltaBiases = np.zeros(len(self.neurons))
            print("cccc")



        self.rawOutputs = (np.matmul(self.weights,((np.array(self.inputs)[np.newaxis]).T))+(np.array(self.biases)[np.newaxis]).T).flatten()
        self.outputs.clear()
        for i,neuron in enumerate(self.neurons):
            if (Debug):
                print("------Running Neuron"+str(i+1)+"------")
                print("Inputs:",self.inputs)
                print("Weights:",self.weights[i])
                print("Bias:", self.biases[i])

            neuron.setRawOutput(self.rawOutputs[i])
            neuron.setOutput(AF.ActivationFunction.runActivationFunction(neuron.getActivationName(),self.rawOutputs,i))

            if (Debug):
                print("Raw Output:",neuron.getRawOutput())
                print("Output:",neuron.getOutput())

            self.outputs.append(self.neurons[i].getOutput())



    def changeWeight(self,indexs,weight):
        self.weights[indexs[0]][indexs[1]] = weight

    def setWeights(self,weights):
        self.weights = weights

    def getWeights(self):
        return self.weights

    def changeNewWeight(self,indexs,weight):
        self.newWeights[indexs[0]][indexs[1]] = weight

    def setNewWeights(self,newWeights):
        self.newWeights = newWeights

    def getNewWeights(self):
        return self.newWeights

    def changeDeltaWeight(self,indexs,deltaWeight):
        self.deltaWeights[indexs[0]][indexs[1]] = deltaWeight

    def setDeltaWeights(self,deltaWeights):
        self.deltaWeights = deltaWeights

    def getDeltaWeights(self):
        return self.deltaWeights



    def changeBias(self,index,bias):
        self.biases[index] = bias

    def getBiases(self):
        return self.biases

    def setBiases(self,biases):
        self.biases = biases

    def changeNewBias(self,index,bias):
        self.newBiases[index] = bias

    def setNewBiases(self,newBiases):
        self.newBiases = newBiases

    def getNewBiases(self):
        return self.newBiases

    def changeDeltaBias(self,index,deltaBias):
        self.deltaBiases[index] = deltaBias

    def setDeltaBiases(self,deltaBiases):
        self.deltaBiases = deltaBiases

    def getDeltaBiases(self):
        return self.deltaBiases



    def setInputs(self,liste): #multiple change
        self.inputs=liste

    def getInputs(self):
        return self.inputs

    def getRawOutputs(self):
        return self.rawOutputs

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

    def getNeuron(self,index):
        return self.neurons[index]

    def getNeuronCount(self):
        return len(self.neurons)

    def getOutputs(self):
        return self.outputs

    def getActivationNames(self):
        names=list()
        for i in range(len(self.neurons)):
            names.append(self.neurons[i].getActivationName())
        return names

    def createRandomWeights(self):
        #self.weights = np.random.rand(len(self.neurons),len(self.inputs))
        self.weights = np.ones((len(self.neurons),len(self.inputs)))/10

    def createRandomBias(self):
        #self.biases = np.random.rand(len(self.neurons))
        self.biases = np.ones(len(self.neurons))/10

    def getLocalDerivatives(self):
        derivatives = list()
        for i in range(len(self.neurons)):
            derivatives.append(self.neurons[i].getLocalDerivative())
        return derivatives

    def setLocalDerivatives(self,derivatives):
        for i in range(len(self.neurons)):
            self.neurons[i].setLocalDerivative(derivatives[i])

    def getAFDerivatives(self):
        derivatives = list()
        for i, neuron in enumerate(self.getNeurons()):
            derivatives.append(AF.ActivationFunction.runActivationFunctionDerivative(neuron.getActivationName(),self.getRawOutputs(), i))
        return derivatives

