

def runBackward(Model):
    for layer in Model.layers:
        for idx,neuron in enumerate(layer.neurons):
            cross_entropy_error_derivative(Model.ExpectedOutputs[idx], neuron.getOutput()) #a(E/y1)
            runActivationFunctionDerivative(neuron.getActivationName(), neuron.getRawOutput()) # a(y1/y1'
            for weight in neuron.weights:
