import CreateModel as cm

model = cm.Model(3136,10,"Softmax","MeansSquaredError")
model.insertHiddenLayer(model.createLayer(128, "ReLU"))
model.getOutputLayer().getNeuron(0).setWeights([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# layer = deneme.createLayer(10, ["ReLU","ReLU","ReLU","Softmax","Softmax","Softmax","ReLU","ReLU","ReLU","Ali"])
# deneme.changeOutputLayer(layer)
model.getModelInfo()

x_values=[0]*3136
y_values= [1,0,0,0,0,0,0,0,0,0]

model.fit(x_values,y_values,1,Debug=False)
print("--------------------Outputs--------------------")
print(model.getOutputs())
print("---------------------Error---------------------")
print(model.getError()[0])