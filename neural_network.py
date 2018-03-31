# encoding : utf-8

import numpy as np

# Loss functions
class LossFunction:
	def apply(self, Ypred, Y):
		print("abstract method")
	def apply_derivative(self, Ypred, Y):
		print("abstract method")


class L_MSE(LossFunction):
	def apply(self, Ypred, Y):
		return np.square(Y - Ypred).sum() / 2
	
	def apply_derivative(self, Ypred, Y):
		return Ypred - Y


class L_SoftMax(LossFunction):
	def apply(self, Ypred, Y):
		e = np.exp( np.max(Ypred*Y, axis=1)) / np.sum(np.exp(Ypred), axis=1)
		L = np.mean(-np.log(e))
		return L

	def apply_derivative(self, Ypred, Y):
		e = np.exp(Ypred*Y) / np.sum(np.exp(Ypred), axis=0)
		return e - Y


# Activation functions
class ActivationFunction:
	def apply(self, I):
		print("abstract method")
	
	def apply_derivative(self, loss):
		print("abstract method")


class AF_Sigmoid(ActivationFunction):
	def apply(self, I):
		return 1 / (1 + np.exp(-I))
	
	def apply_derivative(self, output):
		return output * (1 - output)


class AF_ReLU(ActivationFunction):
	def apply(self, I):
		return np.maximum(0, I)
	
	def apply_derivative(self, output):
		d = np.array(output > 0, dtype=float)
		return d



# Neural Network
class Layer:
	def __init__(self, input_size, output_size, activation_func, learning_rate, batch_normalization=False):
		self.input_size = input_size
		self.ouptut_size = output_size
		
		self.activation_func = activation_func
		self.learning_rate = learning_rate

		self.batch_normalization = batch_normalization
		# weights and biases
		self.O = None
		self.I = None
		self.W = 2 * np.random.random((input_size, output_size)) - 1
		self.B = np.zeros((1, output_size))
	
	def forward_propagation(self, I):
		self.I = I
		O = I.dot(self.W) + self.B
		# normalization
		if self.batch_normalization:
			O = (O - np.mean(O, axis=0)) / np.sqrt(np.var(O, axis=0))
		self.O = self.activation_func.apply(O)
		
		return self.O

	def back_propagation(self, loss):
		d = loss * self.activation_func.apply_derivative(self.O)
		self.update_weights(d)
		return d.dot(self.W.T)
	
	def update_weights(self, delta):
		self.W -= self.learning_rate * self.I.T.dot(delta)
		self.B = self.B - self.learning_rate * delta

class Network:
	def __init__(self, layers_dim, activation_func, loss_func, learning_rate, batch_normalization=False):
		self.activation_func = activation_func
		self.learning_rate = learning_rate
		self.batch_normalization = batch_normalization
		self.loss_func = loss_func
		self.layers = []
		
		self.create_layers(layers_dim)

	def create_layers(self, layers_dim):
		for i in range(len(layers_dim) - 1):
			self.layers.append(Layer(layers_dim[i], layers_dim[i+1], self.activation_func, self.learning_rate, self.batch_normalization))

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward_propagation(self, X):
		Ypred = X
		for layer in self.layers:
			Ypred = layer.forward_propagation(Ypred)
		return Ypred

	def back_propagation(self, Ypred, Y):
		curr_grad = self.loss_func.apply_derivative(Ypred, Y)


		rev_layers = list(self.layers)
		rev_layers.reverse()
		for layer in rev_layers:
			curr_grad = layer.back_propagation(curr_grad)
			
		return curr_grad

	def compute_loss(self, Ypred, Y):
		return self.loss_func.apply(Ypred, Y)

