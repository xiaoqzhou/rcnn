import theano.tensor as T
def ReLU(x):
    """
    Activation function ReLu (Rectified Linear Units)
    """
    y = T.maximum(0.0, x)
    return y
