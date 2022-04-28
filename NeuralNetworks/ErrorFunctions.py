import numpy as np
class ErrorFunctions:
    function_names = ["MeansSquaredError", "CrossEntropyError"]

    @staticmethod
    def getFunctionName():
        return ErrorFunctions.function_name

    @staticmethod
    def mean_squared_error(y, t):
        return 0.5 * ((y-t)**2)

    @staticmethod
    def mean_squared_error_derivative(y, t):
        return t - y

    @staticmethod
    def cross_entropy_error(y, t):
        y=np.array(y)
        t=np.array(t)
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y)) / batch_size

    def cross_entropy_error_derivative(y, t):
        return y - t

    @staticmethod
    def calculateErrors(function_name,y, t):
        if not(function_name in ErrorFunctions.function_names):
            raise ValueError("ErrorFunction: function_name must be one of the following: " + str(ErrorFunctions.function_names))

        if function_name == "MeansSquaredError":
            errors = list()
            for i in range(len(y)):
                errors.append(ErrorFunctions.mean_squared_error(y[i], t[i]))
            return errors
        # elif function_name == "CrossEntropyError":           Not implemented yet
        #     return ErrorFunctions.cross_entropy_error(y, t)