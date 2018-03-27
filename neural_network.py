# encoding : utf-8

import numpy as np

# Loss functions
class LossFunction:
	def apply(Ypred, Y):
		print("abstract method")
	def apply_derivative(Ypred, Y):
		print("abstract method")

class L_MSE(LossFunction):
	def apply(Ypred, Y):
		return np.square(Y - Ypred).sum() / 2
	def apply_derivative(Ypred, Y):
		return Y - Ypred



# Activation functions
class ActivationFunction:
	def apply(I):
		print("abstract method")
	
	def apply_derivative(loss):
		print("abstract method")


class AF_Sigmoid(ActivationFunction):
	def apply(I):
		return 1 / (1 + np.exp(-I))
	
	def apply_derivative(output):
		return output * (1 - output)




# Neural Network
class Layer:
	def __init__(self, input_size, output_size, activation_func, learning_rate):
		self.input_size = input_size
		self.ouptut_size = output_size
		
		self.activation_func = activation_func
		self.learning_rate = learning_rate
		# weights and biases
		self.O = None
		self.I = None
		self.W = 2 * np.random.random((input_size, output_size)) - 1
		self.B = np.zeros((1, output_size))
	
	def forward_propagation(self, I):
		self.I = I
		O = I.dot(self.W) + self.B
		self.O = self.activation_func.apply(O)
		return self.O

	def back_propagation(self, loss):
		d = loss * self.activation_func.apply_derivative(self.O)
		self.update_weights(self.learning_rate, d)
		return d.dot(self.W.T)
	
	def update_weights(self, learning_rate, delta):
		self.W += learning_rate * self.I.T.dot(delta)		

class Network:
	def __init__(self, layers_dim, activation_func, loss_func, learning_rate):
		self.activation_func = activation_func
		self.learning_rate = learning_rate
		self.loss_func = loss_func
		self.layers = []
		
		self.create_layers(layers_dim)

	def create_layers(self, layers_dim):
		for i in range(len(layers_dim) - 1):
			self.layers.append(Layer(layers_dim[i], layers_dim[i+1], self.activation_func, self.learning_rate))

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
