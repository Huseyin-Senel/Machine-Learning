import numpy as np

import CreateModel as cm

model = cm.Model(3136,10,"Softmax","MeansSquaredError")
model.insertHiddenLayer(model.createLayer(128, "ReLU"))

# model.getOutputLayer().getNeuron(0).setWeights([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
# layer = deneme.createLayer(10, ["ReLU","ReLU","ReLU","Softmax","Softmax","Softmax","ReLU","ReLU","ReLU","Ali"])
# deneme.changeOutputLayer(layer)

model.getModelInfo()


x_values=np.zeros((1, 3136))
y_values=np.zeros((1, 10))

x_value=[0]*3136
y_value= [1,0,0,0,0,0,0,0,0,0]
x_values[0]=x_value
y_values[0]=y_value

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[1]*3136
y_value1=[0,1,0,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)

x_value1=[2]*3136
y_value1=[0,0,1,0,0,0,0,0,0,0]
x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)


model.fit(x_values,y_values,10,0.9,Debug=False)


# model = cm.Model(2,10,"Softmax","MeansSquaredError")
# model.insertHiddenLayer(model.createLayer(3, "ReLU"))
# model.getModelInfo()
#
# x_values=np.zeros((1, 3))
# y_values=np.zeros((1, 10))
#
# x_value=[0]*3
# y_value= [1,0,0,0,0,0,0,0,0,0]
# x_values[0]=x_value
# y_values[0]=y_value
#
# x_value1=[1]*3
# y_value1=[0,1,0,0,0,0,0,0,0,0]
# x_values=np.append(np.array(x_values),[np.array(x_value1)],axis=0)
# y_values=np.append(np.array(y_values),[np.array(y_value1)],axis=0)
#
# model.fit(x_values,y_values,40,0.9,Debug=False)