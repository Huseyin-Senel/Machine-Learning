import random
import ActivationFunction as AF

class neuron:
    current_activation=-1
    X_Values=list()
    X_Values.append(1)

    nY_Value=0             #the value which didnt enter activationFunction (pure output)
    Y_Value=0              #this value entered activation function


    def __init__(self,*args):

        if len(args) == 2 and isinstance(args[0],list):
         #args[0] = weights args[1] = activation_function
            self.Weights=args[0]
            self.af_obj=AF.ActivationFunction(args[1])                      #we Created af_obj in constructor
            self.af_name= self.af_obj.getActivationFunction()             #also created af_name in constructor
         #args[0] = weightsCount args[1] = ActivationFunction()
        elif len(args) == 2 and isinstance(args[0],int):
            self.Weights = [0]*args[0]                        #we created list which is from zeros
            self.createRandomWeights(args[0])
            self.af_obj = AF.ActivationFunction(args[1])  # we Created af_obj in constructor
            self.af_name = self.af_obj.getActivationFunction()  # also created af_name in constructor

    def changeInputs(self,liste):  #çoklu değişim
        self.X_Values[1:]=liste
        self.createRandomWeights(len(self.X_Values))

    def changeWeights(self, liste):
        self.Weights=liste

    def changeInput(self,index,value): #tek değişim yapıyor
        self.X_Values[index]=value

    def changeWeight(self,index,value): #tek değişim yapıyor
        self.Weights[index]=value

    def runNeuron(self):
        print("X",self.X_Values)
        print("weights: ",self.Weights)
        for idx,i in enumerate(self.X_Values):
            self.nY_Value+= (i*self.Weights[idx])

        self.Y_Value = self.af_obj.runActivationFunction(self.af_name,self.nY_Value)

    def getOutput(self):
        return self.Y_Value

    def getActivationName(self):
        return self.af_name

    def getNOutput(self):
        return self.nY_Value

    def getWeights(self):
        return self.Weights

    def getInputs(self):
        return self.X_Values

    def createRandomWeights(self,weights_number):
        self.Weights = [0] * weights_number # we created list which is from zeros
        for i in range(weights_number):
            self.Weights[i] = random.random()