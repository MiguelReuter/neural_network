# encoding : utf-8

import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

from neural_network import *


def lecture_mnist(data_path):
	mdata = MNIST(data_path)
	data_app, label_app = mdata.load_training()
	data_test, label_test = mdata.load_testing()

	return np.array(data_app), np.array(data_test), np.array(label_app), np.array(label_test)

def pick_batches(n, data, label):
	_xr = np.arange(data.shape[0])
	np.random.shuffle(_xr)
	idx = _xr[:n]

	x = data[idx, :]
	y_cl = label[idx]

	y = np.zeros((n, 10))
	for i in range(n):
		y[i, y_cl[i]] = 1

	x = preprocess_data(x)

	return x, y, y_cl

def preprocess_data(x):
	return (x.T - np.mean(x, axis=1)).T



if __name__ == "__main__":
	np.random.seed(1)
	# pour que l'exécution soit déterministe
	##########################
	# Génération des données #
	##########################
	# N est le nombre de données d'entrée
	# D_in est la dimension des données d'entrée
	# D_h le nombre de neurones de la couche cachée
	# D_out est la dimension de sortie (nombre de neurones de la couche de sort
	
	"""
	N, D_in, D_h, D_out = 30, 2, 10, 3
	learning_rate = 0.1

	#custom_sig = AF_ReLU()
	custom_sig = AF_Sigmoid()
	mse = L_MSE()
	#mse = L_SoftMax()
	
	losses = []

	# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
	X = np.random.random((N, D_in))
	Y = np.random.random((N, D_out))

	nn = Network([D_in, D_h, D_out], custom_sig, mse, learning_rate)
	for i in range(epochs_max):
		Ypred = nn.forward_propagation(X)
		loss = nn.compute_loss(Ypred, Y)
		losses.append(loss)
		if i % 1000 == 0:		
			print("iteration {} : ".format(i), loss)
		nn.back_propagation(Ypred, Y)
	"""
	data_app, data_test, label_app, label_test	= lecture_mnist("../mnist-data")

	N = 64
	epochs_max = 300
	
	# N (28*28) px images - 10 classes
	D_in, D_out = 28*28, 10

	hidden_layers_dim = [50]
	learning_rate = 0.00005

	af_func = AF_ReLU()
	af_func = AF_Sigmoid()
	#loss_func = L_MSE()
	loss_func = L_SoftMax()

	nn = Network([D_in, *hidden_layers_dim, D_out], af_func, loss_func, learning_rate, batch_normalization=True)
	
	# train
	losses = []
	for epoch in range(epochs_max):
		X, Y, Y_label = pick_batches(N, data_app, label_app)

		Ypred = nn.forward_propagation(X)
		loss = nn.compute_loss(Ypred, Y)
		nn.back_propagation(Ypred, Y)
		
		losses.append(loss)
	
	# test
	X, Y, Y_label = pick_batches(N, data_test, label_test)
	Ypred = nn.forward_propagation(X)
	y_pred_cl = np.argmax(Ypred, axis=1)
	acc = np.mean(np.array(y_pred_cl == Y_label, dtype=int))
	print("acc = ", acc)

	# plot loss(epoch)
	fig = plt.figure()	
	plt.plot(losses)
	plt.axis([0, epochs_max, 0, max(losses) + 1])
	plt.draw()
	plt.pause(1)
	print(loss)
	input("PRESS ANY KEY TO CONTINUE.")
	plt.close(fig)














	
