import numpy as np
from alive_progress import alive_bar
import neuron as NU
import LayerClass as LC
import ErrorFunctions as EF
import ActivationFunction as AF
class Model:


    def __init__(self,inputCount,outputCount,activationFunction,errorFunction):
        self.layers = list()
        self.inputs = list()
        self.ExpectedOutputs = list()
        self.outputCount = outputCount
        self.inputCount = inputCount
        self.ErrorFunction = errorFunction
        self.Errors = list()
        self.TotalError = 0

        self.insertHiddenLayer(self.createLayer(outputCount, activationFunction))

    def runModel(self,learningRate,Debug=False):
        for idx,layer in enumerate(self.layers):
            if idx==0:
                layer.setInputs(self.inputs)
            else:
                layer.setInputs(self.layers[idx-1].getOutputs())
            if (Debug): print("\n-------------------------- Running Layer"+str(idx+1)+"----------------------------------------")
            layer.runLayer(Debug)
        self.calculateErrors()
        self.TotalError = sum(self.Errors) if not(self.Errors == None) else 0
        if (Debug):print("\n------------Outputs-------------")
        if (Debug):print("Total Error:",self.TotalError)
        self.runBackPropagation(learningRate)
        # print ("Outputs:",(self.layers[-1].getOutputs()))
        if (Debug):print (["%0.2f" % i for i in self.layers[-1].getOutputs()])
        if (Debug):print("Expected Outputs:",self.ExpectedOutputs)

    def runBackPropagation(self,learningRate):
        for layer in reversed(self.layers):
            layer.setNewWeights(np.zeros(layer.getWeights().shape))
            layer.setNewBiases(np.zeros(layer.getBiases().shape))
            if layer == self.layers[-1]:
                for i, neuron in enumerate(self.layers[-1].getNeurons()):
                    EF_Derivative = EF.ErrorFunctions.calculateDerivatives(self.ErrorFunction, self.ExpectedOutputs[i],neuron.getOutput())

                    if neuron.getActivationType() =="AF":
                        AF_Derivative = AF.ActivationFunction.runActivationFunctionDerivative(neuron.getActivationName(), neuron.getOutput())
                    elif neuron.getActivationType() =="LAF":
                        AF_Derivative = AF.ActivationFunction.runActivationFunctionDerivative(neuron.getActivationName(), neuron.getOutput(),layer.getOutputs())[i]

                    neuron.setLocalDerivative(EF_Derivative * AF_Derivative)

                    for idx,x in enumerate(layer.getInputs()):
                        layer.changeNewWeight([i,idx],layer.getWeights()[i][idx] - learningRate * neuron.getLocalDerivative() * x)
                    layer.changeNewBias(i,layer.getBiases()[i] - learningRate * neuron.getLocalDerivative())
            else:
                # self.layers[self.layers.index(layer) + 1].setWeights(self.layers[self.layers.index(layer) + 1].getNewWeights())
                # self.layers[self.layers.index(layer) + 1].setBiases(self.layers[self.layers.index(layer) + 1].getNewBiases())
                for i, neuron in enumerate(layer.getNeurons()):
                    c = 0
                    for j, neuronL in enumerate(self.layers[self.layers.index(layer) + 1].getNeurons()):
                        c += neuronL.getLocalDerivative() * self.layers[self.layers.index(layer) + 1].getWeights()[j][i]

                    AF_Derivative = AF.ActivationFunction.runActivationFunctionDerivative(neuron.getActivationName(), neuron.getOutput())
                    neuron.setLocalDerivative(c * AF_Derivative)

                    for idx, x in enumerate(layer.getInputs()):
                        layer.changeNewWeight([i,idx],layer.getWeights()[i][idx] - learningRate * neuron.getLocalDerivative() * x)
                    layer.changeNewBias(i,layer.getBiases()[i] - learningRate * neuron.getLocalDerivative())
        # self.layers[0].setWeights(self.layers[0].getNewWeights())
        # self.layers[0].setBiases(self.layers[0].getNewBiases())
        for layer in self.layers:
            layer.setWeights(layer.getNewWeights())
            layer.setBiases(layer.getNewBiases())




    def calculateErrors(self):
        self.Errors=EF.ErrorFunctions.calculateErrors(self.ErrorFunction,self.ExpectedOutputs,self.layers[-1].getOutputs())

    def fit(self,x_train,y_train,epochs,LearningRate,Debug=False):
        for epoch in range(epochs):
            with alive_bar(len(x_train),title="Epoch: "+str(epoch+1),force_tty=True) as bar:
                error = 0
                for i, x in enumerate(x_train):
                    self.setInputs(x)
                    self.setExpectedOutputs(y_train[i])
                    self.runModel(LearningRate, Debug)
                    error += self.getError()[0]
                    bar()
            print("Error: " + str(error / len(x_train)))




    def setExpectedOutputs(self,outputs):
        self.ExpectedOutputs=outputs

    def setInputs(self,inputs):
        self.inputs=inputs

    def changeOutputLayer(self,layer):
        self.layers[-1]=layer

    def createLayer(self,neuronCount,activasionFunctions):
        layerObject = LC.Layer(neuronCount,activasionFunctions)
        return layerObject

    def insertHiddenLayer(self, layer):
        if len(self.layers)==0:
            self.layers.append(layer)
        else:
            self.layers.insert(-1, layer)

    def changeLayer(self,index,layer):
        self.layers[index]=layer

    def createNeuron(self,activationFunction,Weights):
        neuronObject = NU.Neuron(activationFunction,Weights)
        return neuronObject

    def getOutputs(self):
        return self.layers[-1].getOutputs()

    def getLayers(self):
        return self.layers

    def getTotalNeurons(self):
        totalNeurons=0
        for layer in self.layers:
            totalNeurons+=layer.getNeuronCount()
        return totalNeurons

    def getConnectionCount(self,layerIndex):
        if layerIndex==0:
            return self.inputCount*self.layers[0].getNeuronCount()
        else:
            return self.layers[layerIndex-1].getNeuronCount()*self.layers[layerIndex].getNeuronCount()

    def totalConnectionCount(self):
        totalConnections=0
        for i in range(len(self.layers)):
            totalConnections+=self.getConnectionCount(i)+self.layers[i].getNeuronCount()
        return totalConnections

    def getOutputLayer(self):
        return self.layers[-1]

    def getLayer(self,index):
        return self.layers[index]

    def getError(self):
        return self.TotalError , self.Errors

    def getModelInfo(self):
        print("\n<-- Model Info: -->\n ")
        print("Input Count: ",self.inputCount)
        print("Output Count: ",self.outputCount)
        print("Total Layers Count: ",len(self.layers))
        print("   Hidden Layers: ",len(self.layers)-1)
        print("   Output Layer:  1")
        print("Total Neurons Count: ",self.getTotalNeurons())
        for idx,layer in enumerate(self.layers):
            print("   Layer"+str(idx+1)+": "+str(layer.getNeuronCount()))
        print("Total number of connections",self.totalConnectionCount())
        for i in range(len(self.layers)):
            print("   Layer"+str(i+1)+": "+str(self.getConnectionCount(i))+" + "+str(len(self.layers[i].getNeurons())))
        print("\n<-- End of Model Info -->\n")
