import neuron as NU
class Layer:
    def __init__(self,*args): # args[0]=neuron Count     args[1] = activation Function list OR just activation Function
        self.neurons=list()
        self.inputs=list()   #N input
        self.outputs=list()  #M output NXM
        # args[0] =Neuron Numbers , args[1] = Activation Function
        if len(args) == 1 and isinstance(args[0], int):
            for i in range(args[0]):
                self.neurons.append(NU.neuron(len(self.inputs),"ReLU"))

        elif len(args) == 2 and isinstance(args[1], str):
            for i in range(args[0]):
                self.neurons.append(NU.neuron(len(self.inputs), args[1]))

        elif len(args) == 2 and isinstance(args[1], list):
            for i in range(args[0]):
                self.neurons.append(NU.neuron(len(self.inputs), args[1][i]))

    def addNeuron(self,neuron):
        self.neurons.append(neuron)

    def dropNeuron(self,*args):
        if len(args) == 0:
            del self.neurons[-1]

        elif len(args) == 1 and isinstance(args[0], int):
            self.neurons.pop(args[0])

    def getNeurons(self):
        return self.neurons

    def runLayer(self):
        self.outputs.clear()
        for i in range(len(self.neurons)):
            self.neurons[i].changeInputs(self.inputs)
            self.neurons[i].runNeuron()
            self.outputs.append(self.neurons[i].getOutput())


    def changeInputs(self,liste): #multiple change
        self.inputs=liste

    def changeInput(self,value,index): #it changes only one input
        self.inputs[index] = value

    def getOutputs(self):
        return self.outputs



layer1= Layer(4,"Sigmoid") #if we dont give weights it give automatically
layer1.changeInputs([1,2])
layer1.runLayer()

print("")
print("############ LAYER2 YE GEÇİLDİ ###############")                                            #burda 2. bir katman ekledik araya
print("")

layer2 = Layer(2,"Sigmoid")
layer2.changeInputs(layer1.outputs)
layer2.runLayer()

print("")
print("############ SONUÇLAR ###############")
print("")

print(layer2.getOutputs())

#Burda çıktılarda ilk(ilk katman) 4 X değerine hepsine random weights değeri atandı çünkü biz elle vermedik değerleri
#sonra bu değerlerden hesap edilen sonuçlar 2. katmandaki inputlara verildi ve 2 neuronada random weightsler atandı
#bu değerler ile hesaplanan değerler çıkışa verildi

