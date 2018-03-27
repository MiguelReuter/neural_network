# encoding : utf-8

import numpy as np


class LossFunction:
	pass



class ActivationFunction:
	def apply_function(I):
		pass
	
	def apply_derivative(I):
		pass


class Layer:
	def __init__(self, input_size, output_size, activation_func)
		self.input_size = input_size
		self.ouptut_size = output_size
		self.activation_func = activation_func
		# weights and biases
		self.W = 2 * np.random.random((input_size, output_size)) - 1
		self.B = np.zeros((output_size))
	
	def forward_propagation(I):
		return I.dot(self.W) + self.B

	def back_propagation(O):
		pass
		

class Network:
	pass
