import ActivationFunction as AF

class neuron:

    def __init__(self,*args):
        self.af_name = "ReLU"
        self.rawOutput = 0  # the value which didnt enter activationFunction (pure output)
        self.output = 0  # this value entered activation function
        self.LocalDerivative = None

        if len(args) == 1 and isinstance(args[0],str):    # args[0] = activation_function
            self.af_name = AF.ActivationFunction.checkActivationFunction(args[0])

    def setOutput(self,value):
        self.output=value

    def getOutput(self):
        return self.output

    def getRawOutput(self):
        return self.rawOutput

    def setRawOutput(self,value):
        self.rawOutput=value

    def getActivationName(self):
        return self.af_name

    def getActivationType(self):
        return self.af_type

    def setActivationFunction(self,af_name):
        self.af_name,self.af_type = AF.ActivationFunction.checkActivationFunction(af_name)

    def setLocalDerivative(self,value):
        self.LocalDerivative = value

    def getLocalDerivative(self):
        return self.LocalDerivative