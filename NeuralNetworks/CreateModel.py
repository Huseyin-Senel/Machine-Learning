import neuron as NU
import LayerClass as LC
class Model:


    def __init__(self,inputCount,outputCount,activationFunction):
        self.layers = list()
        self.inputs = list()

        self.outputCount = outputCount
        self.inputs=[0]*inputCount
        self.insertHiddenLayer(self.createLayer(outputCount, activationFunction))

    def runModel(self,Debug=False):
        for idx,layer in enumerate(self.layers):
            if idx==0:
                layer.setInputs(self.inputs)
            else:
                layer.setInputs(self.layers[idx-1].getOutputs())
            if Debug : print("\n-------------------------- Running Layer"+str(idx+1)+"----------------------------------------")
            layer.runLayer(Debug)

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
            return len(self.inputs)*self.layers[0].getNeuronCount()
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

    def getModelInfo(self):
        print("\n<-- Model Info: -->\n ")
        print("Input Count: ",len(self.inputs))
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





# deneme = Model(3136,10,"Softmax")
# deneme.insertHiddenLayer(deneme.createLayer(128, "ReLU"))
#
# deneme.getOutputLayer().getNeuron(0).setWeights([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#
# # layer = deneme.createLayer(10, ["ReLU","ReLU","ReLU","Softmax","Softmax","Softmax","ReLU","ReLU","ReLU","Ali"])
# # deneme.changeOutputLayer(layer)
#
# deneme.getModelInfo()
#
# list=[0]*3136
# deneme.setInputs(list)
# #
# deneme.runModel(Debug=True)
# print("--------------------Outputs--------------------")
# print(deneme.getOutputs())
