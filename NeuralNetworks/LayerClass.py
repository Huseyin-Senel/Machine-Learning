import neuron as NU
import ActivationFunction as AF

class Layer:


    def __init__(self,*args):

        self.neurons = list()
        self.inputs = list()  # N input
        self.outputs = list()  # M output NXM

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

    def setInputs(self,liste): #multiple change
        self.inputs=liste

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

    def runLayer(self,Debug=False):

        RawOutputs = list()
        self.outputs.clear()
        for i in range(len(self.neurons)):
            self.neurons[i].setInputs(self.inputs)
            if Debug : print("------Running Neuron"+str(i+1)+"------")
            self.neurons[i].runNeuron(Debug)
            self.outputs.append(self.neurons[i].getOutput())
            RawOutputs.append(self.neurons[i].getRawOutput())

        if Debug :print("---------------Running Layer Activation Function-------------")
        for idx,neuron in enumerate(self.neurons):
            self.outputs[idx]=AF.ActivationFunction.runActivationFunctionLayer(neuron.getActivationName(),RawOutputs)[idx]
        if Debug: print("Layer Outputs:",self.outputs)