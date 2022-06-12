import numpy as np
import CreateModel as cm
import matplotlib.pyplot as plt

# from keras.datasets import mnist
# from keras.utils.np_utils import to_categorical
#
# (X_train, y_train),(X_test, y_test) = mnist.load_data()
#
# y_cat_test = to_categorical(y_test,10)
# y_cat_train = to_categorical(y_train,10)
# X_train = X_train/X_train.max()
# X_test = X_test/X_test.max()
# X_train = X_train.reshape(60000, 28, 28, 1)
# X_test = X_test.reshape(10000, 28, 28, 1)
#
# #flat X_train
# X_train = X_train.reshape(60000, 28*28)
# X_test = X_test.reshape(10000, 28*28)


# Create dataset------------------------------------------------------------------
X = np.array([0])
Y = np.array([1,0,0,0,0,0,0,0,0,0])
for i in range(0,100):
    X = np.vstack([X,np.array([0])])
    Y = np.vstack([Y,np.array([1,0,0,0,0,0,0,0,0,0])])
    X = np.vstack([X,np.array([0.1])])
    Y = np.vstack([Y,np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])])
    X = np.vstack([X, np.array([0.2])])
    Y = np.vstack([Y, np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])])
    X = np.vstack([X, np.array([0.3])])
    Y = np.vstack([Y, np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])])
    X = np.vstack([X, np.array([0.4])])
    Y = np.vstack([Y, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])])
    X = np.vstack([X, np.array([0.5])])
    Y = np.vstack([Y, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])])
    X = np.vstack([X, np.array([0.6])])
    Y = np.vstack([Y, np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])])
    X = np.vstack([X, np.array([0.7])])
    Y = np.vstack([Y, np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])])
    X = np.vstack([X, np.array([0.8])])
    Y = np.vstack([Y, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])])
    X = np.vstack([X, np.array([0.9])])
    Y = np.vstack([Y, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])])
#----------------------------------------------------------------------------------------


# Create model---------------------------------------------------------------------------
model = cm.Model(1,10,"Softmax")  #input_size, output_size, activation_function(for the output layer)
model.insertHiddenLayer(model.createLayer(10, "Sigmoid"))   #neuron_count, activation_function
# model.insertHiddenLayer(model.createLayer(64, "ReLU"))
# model.insertHiddenLayer(model.createLayer(5, "ReLU","ReLU","ReLU","Softmax","Sigmoid"))
# layer = model.createLayer(10, "Softmax")
# model.changeOutputLayer(layer)
model.getModelInfo()
#----------------------------------------------------------------------------------------


# Train model----------------------------------------------------------------------------
model.fit(X,Y,Epoch=5000,LearningRate=0.6,momentum=0.3,ErrorFunction="MeansSquaredError",interface=False,Debug=False)

# print(model.getModel()[0]) #print model weights
# print(model.getModel()[1]) #print model biases
#----------------------------------------------------------------------------------------

# Test model-----------------------------------------------------------------------------
# print(model.predict(X[0])) #print example prediction
#----------------------------------------------------------------------------------------



# Plot model-----------------------------------------------------------------------------
data = model.getHistory()

arrayData = np.array(data)

file = open("sample.txt", "a")
content = str(arrayData)
file.write(content+"\n\n")
file.close()

plt.plot(data, color='red') #plot the data
plt.xticks(range(0,len(data)+1, 500)) #set the tick frequency on x-axis

plt.ylabel('Error') #set the label for y axis
plt.xlabel('Epoch') #set the label for x-axis
plt.title("Accuracy") #set the title of the graph
plt.show() #display the graph
#----------------------------------------------------------------------------------------