import neuron as NU
import LayerClass as LC
import ErrorFunctions as EF
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

    def runModel(self,Debug=False):
        for idx,layer in enumerate(self.layers):
            if idx==0:
                layer.setInputs(self.inputs)
            else:
                layer.setInputs(self.layers[idx-1].getOutputs())
            if Debug : print("\n-------------------------- Running Layer"+str(idx+1)+"----------------------------------------")
            layer.runLayer(Debug)
        self.calculateErrors()
        self.TotalError = sum(self.Errors) if not(self.Errors == None) else 0

    def calculateErrors(self):
        self.Errors=EF.ErrorFunctions.calculateErrors(self.ErrorFunction,self.ExpectedOutputs,self.layers[-1].getOutputs())

    def fit(self,x_train,y_train,epochs,Debug=False):
        self.setInputs(x_train)
        self.setExpectedOutputs(y_train)
        for epoch in range(epochs):
            self.runModel(Debug)

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
            totalConnections+=self.getConnectionCount(i)
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
            print("   Layer"+str(i+1)+": "+str(self.getConnectionCount(i)))
        print("\n<-- End of Model Info -->\n")
