import numpy as np
class ErrorFunctions:
    function_names = ["MeansSquaredError","CrossEntropyError"]


    @staticmethod
    def mean_squared_error(y, t):   #Y true result  T NN result
        return 0.5 * ((y-t)**2)

    @staticmethod
    def mean_squared_error_derivative(y, t):
        return t - y

    @staticmethod
    def cross_entropy(y, t):
        return -np.sum(t * np.log(y))

    @staticmethod
    def cross_entropy_derivative(y, t):
        return -t / y


    @staticmethod
    def calculateErrors(function_name,y, t):
        if not(function_name in ErrorFunctions.function_names):
            raise ValueError("ErrorFunction: function_name must be one of the following: " + str(ErrorFunctions.function_names))

        if function_name == ErrorFunctions.function_names[0]:
            errors = list()
            for i in range(len(y)):
                errors.append(ErrorFunctions.mean_squared_error(y[i], t[i]))
            return errors

        elif function_name == ErrorFunctions.function_names[1]:
            errors = list()
            for i in range(len(y)):
                errors.append(ErrorFunctions.cross_entropy(y, t))
            return errors


    @staticmethod
    def calculateDerivatives(function_name,y, t):
        if function_name == ErrorFunctions.function_names[0]:
            return ErrorFunctions.mean_squared_error_derivative(y, t)
        elif function_name == ErrorFunctions.function_names[1]:
            return ErrorFunctions.cross_entropy_derivative(y, t)
