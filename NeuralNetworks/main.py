import CreateModel as cm

model = cm.Model(3136,10,"Softmax")
model.insertHiddenLayer(model.createLayer(128, "ReLU"))
model.getOutputLayer().getNeuron(0).setWeights([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# layer = deneme.createLayer(10, ["ReLU","ReLU","ReLU","Softmax","Softmax","Softmax","ReLU","ReLU","ReLU","Ali"])
# deneme.changeOutputLayer(layer)
model.getModelInfo()

input=[0]*3136
model.setInputs(input)
model.runModel(Debug=False)
print("--------------------Outputs--------------------")
print(model.getOutputs())