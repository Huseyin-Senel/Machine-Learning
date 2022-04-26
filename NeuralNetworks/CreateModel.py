import neuron as NU
import LayerClass as LC
class Model:
    def __init__(self,inputCount,outputCount,activationFunction):
        self.inputCount=inputCount
        self.layers=list()
        self.inputs=[0]*inputCount
        self.addLayer(self.createLayer(outputCount,activationFunction))


    def changeOutputLayer(self,layer):
        self.outputLayer=layer

    def createLayer(self,neuronCount,activasionFunctions):
        layerObject = LC.Layer(neuronCount,activasionFunctions)
        return layerObject

    def addLayer(self,layer):
        self.layers.append(layer)

    def getLayers(self):
        return self.layers

deneme = Model(4,2,"Sigmoid")
print(deneme.getLayers()[0].getNeurons()[0].getActivationName())
