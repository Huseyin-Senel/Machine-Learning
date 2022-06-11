import numpy as np
from alive_progress import alive_bar
import neuron as NU
import LayerClass as LC
import ErrorFunctions as EF
import ActivationFunction as AF
class Model:

    def __init__(self,inputCount,outputCount,activationFunction):
        self.layers = list()
        self.inputs = list()
        self.ExpectedOutputs = list()
        self.outputCount = outputCount
        self.inputCount = inputCount
        self.ErrorFunction = ""
        self.Errors = list()
        self.TotalError = 0
        self.ErrorHistory = list()

        self.insertHiddenLayer(self.createLayer(outputCount, activationFunction))

    def fit(self,x_train,y_train,Epoch,LearningRate,ErrorFunction,interface,Debug=False):
        self.ErrorFunction = ErrorFunction
        for epoch in range(Epoch):
            if not(interface):
                error = 0
                for i, x in enumerate(x_train):
                    self.setInputs(x)
                    self.setExpectedOutputs(y_train[i])
                    self.predictModel(LearningRate,Debug)
                    error += self.getError()[0]
                print("Epoch"+str(epoch+1)+" Error: " + str(error / len(x_train)))
                self.ErrorHistory.append(error / len(x_train))
            else:
                with alive_bar(len(x_train),title="Epoch: "+str(epoch+1),force_tty=True) as bar:
                    error = 0
                    for i, x in enumerate(x_train):
                        self.setInputs(x)
                        self.setExpectedOutputs(y_train[i])
                        self.predictModel(LearningRate, Debug)
                        error += self.getError()[0]
                        bar()
                print("Error: " + str(error / len(x_train)))
                self.ErrorHistory.append(error / len(x_train))

    def predictModel(self,learningRate,Debug):
        for idx,layer in enumerate(self.layers):
            if idx==0:
                layer.setInputs(self.inputs)
            else:
                layer.setInputs(self.layers[idx-1].getOutputs())
            if (Debug): print("\n-------------------------- Running Layer"+str(idx+1)+"----------------------------------------")
            layer.runLayer(Debug)
        self.Errors = self.calculateErrors()
        self.TotalError = sum(self.Errors)
        if (Debug):print("\n----------------------------------Outputs----------------------------------------")
        if (Debug):print ("Layer Outputs:",["%0.2f" % i for i in self.layers[-1].getOutputs()])
        if (Debug):print("Expected Outputs:",self.ExpectedOutputs)
        if (Debug): print("Total Error:", self.TotalError)
        if (Debug): print("-------------------------------------------------------------------------------\n")
        self.runBackPropagation(learningRate,Debug)


    def runBackPropagation(self,learningRate,Debug=False):
        if Debug: print
        for layer in reversed(self.layers):
            if (Debug): print("\n-------------------- Running Backpropagation Layer" + str(self.layers.index(layer)+1) + "-----------------------------")
            layer.setNewWeights(np.zeros(layer.getWeights().shape))
            layer.setNewBiases(np.zeros(layer.getBiases().shape))

            if layer == self.layers[-1]:
                for i, neuron in enumerate(layer.getNeurons()):
                    if (Debug): print("-------- Running Neuron" + str(i + 1) + "-----------")

                    EF_Derivative = EF.ErrorFunctions.calculateDerivatives(self.ErrorFunction, self.ExpectedOutputs[i],neuron.getOutput())  #C   dE/dy
                    if (Debug): print("Error Function Derivative:",EF_Derivative)
                    AF_Derivative = AF.ActivationFunction.runActivationFunctionDerivative(neuron.getActivationName(),layer.getRawOutputs(),i)   #    dy/dy'
                    if (Debug): print("Activation Function Derivative:",AF_Derivative)
                    neuron.setLocalDerivative(EF_Derivative * AF_Derivative)

                    for idx,x in enumerate(layer.getInputs()):
                        deltaWeight = ((-learningRate) * EF_Derivative * AF_Derivative * x)
                        if (Debug): print("New Weight:",layer.getWeights()[i][idx]+deltaWeight)
                        layer.changeNewWeight([i,idx],layer.getWeights()[i][idx] + deltaWeight)
                    deltaBias = ((-learningRate) * EF_Derivative * AF_Derivative)
                    layer.changeNewBias(i,layer.getBiases()[i] + deltaBias)
            else:
                weights = self.layers[self.layers.index(layer) + 1].getWeights()
                if Debug: print("Front weights:\n",weights)
                localDerivatives = np.array(self.layers[self.layers.index(layer) + 1].getLocalDerivatives()).reshape(len(self.layers[self.layers.index(layer) + 1].getLocalDerivatives()), 1)
                if Debug:print("FrontLocalDerivatives:\n",localDerivatives)
                arr1 = np.multiply(weights, localDerivatives)
                if Debug:print("W*FrontLocalDerivatives:\n",arr1)
                arr1 = np.sum(arr1, axis=0)#column sum
                if Debug:print("(W*FrontLocalDerivatives)   sum column:\n",arr1)

                AFderivatives = layer.getAFDerivatives()
                AFderivatives = np.array(AFderivatives).reshape(1, len(AFderivatives))
                if Debug:print("AFderivatives:\n",AFderivatives)

                thisAFLocalDerivates = np.multiply(arr1, AFderivatives)
                if Debug:print("thisAFLocalDerivates:\n",thisAFLocalDerivates)
                layer.setLocalDerivatives(thisAFLocalDerivates.flatten())

                inputs2 = layer.getInputs()
                inputs2 = np.array(inputs2).reshape(1, len(inputs2))
                if Debug:print("inputs:\n",inputs2)
                arr2 = np.multiply(thisAFLocalDerivates.T, inputs2)
                if Debug:print("inputs*(ThisLocalDerivatives.T):\n",arr2)

                if Debug: print("Old Weights:\n", layer.getWeights())
                deltaWeights = (arr2*(-learningRate))
                layer.setNewWeights(layer.getWeights() + deltaWeights)
                if Debug: print("New Weights:\n",layer.getNewWeights())

                deltaBiases =(thisAFLocalDerivates*(-learningRate))
                layer.setNewBiases(layer.getBiases() + deltaBiases)
                if Debug: print("-----------------------------------------------------------------------------------------------")


        for layer in self.layers:
            layer.setWeights(layer.getNewWeights())
            layer.setBiases(layer.getNewBiases())


    def predict(self,input):
        for idx,layer in enumerate(self.layers):
            if idx==0:
                layer.setInputs(input)
            else:
                layer.setInputs(self.layers[idx-1].getOutputs())
            layer.runLayer()
        return self.layers[-1].getOutputs()

    def calculateErrors(self):
        return EF.ErrorFunctions.calculateErrors(self.ErrorFunction,self.ExpectedOutputs,self.layers[-1].getOutputs())

    def setExpectedOutputs(self,outputs):
        self.ExpectedOutputs=outputs

    def setInputs(self,inputs):
        self.inputs=inputs

    def changeOutputLayer(self,layer):
        self.layers[-1]=layer

    def createLayer(self,neuronCount,activationFunctions):
        layerObject = LC.Layer(neuronCount,activationFunctions)
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

    def getModel(self):
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.getWeights())
            biases.append(layer.getBiases())
        return  weights,biases

    def getHistory(self):
        return self.ErrorHistory