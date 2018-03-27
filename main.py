# encoding : utf-8

import numpy as np
from neural_network import *



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
	N, D_in, D_h, D_out = 30, 2, 10, 3
	learning_rate = 0.1
	
	# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
	X = np.random.random((N, D_in))
	Y = np.random.random((N, D_out))

	nn = Network([D_in, D_h, D_out], AF_Sigmoid, L_MSE, learning_rate)
	for i in range(30000):
		Ypred = nn.forward_propagation(X)
		loss = nn.compute_loss(Ypred, Y)
		if i % 1000 == 0:		
			print("iteration {} : ".format(i), loss)
		nn.back_propagation(Ypred, Y)
