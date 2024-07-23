import torch

def softsign(x):
    return x / (1 + x.abs())

def sign(x):
    return x.sign()

def tanh(x):
    return x.tanh()

def sigmoid(x):
    return 2 * x.sigmoid() - 1

def clipped_linear(x):
    return torch.clamp(x, -1, 1)



ACTIVATION = {
    "sign":sign,
    "softsign":softsign,
    "tanh":tanh,
    "sigmoid":sigmoid,
    "linear":clipped_linear,

}