#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           gsd_wr.py
#   author:         Ludovico Fabbri 1197400
#   description:    recommendations using UV decomposition with stocastic gradient descent
#
#
#################################################################################



import csv
import io
import argparse
import math
import numpy as np
import scipy.sparse as sp
from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt


from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
	FileTransferSpeed, FormatLabel, Percentage, \
	ProgressBar, ReverseBar, RotatingMarker, \
	SimpleProgress, Timer

# progress bar settings
widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]



# _N = 671 + 1
# _M = 164979 + 1


_N = 138436 + 1
_M = 131258 + 1


NUM_FOLDS = 1
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"

lmbda = 0.1 # Regularisation weight
k = 20  # Dimension of the latent feature space
n_epochs = 100  # Number of epochs
gamma = 0.01  # Learning rate






def parseOutputFolderPath(filepath):

	output_path = ""
	tokens = filepath.split("/")

	for i in xrange(len(tokens) - 1):
		output_path += tokens[i] + "/"

	return output_path



def parseFileName(filepath):

	tokens = filepath.split("/")
	filename = tokens[len(tokens) - 1]
	tokens = filename.split(".")
	file_name_no_ext = tokens[len(tokens) - 2]
	return file_name_no_ext





def stochasticGradientDescentUV(filepath):

	src_folder = parseOutputFolderPath(filepath)
	base_file_name = parseFileName(filepath)

	avg_rmse = 0.0
	avg_mae = 0.0

	out_file_base = base_file_name + "_pred_sgd_wr_{0}-{1}-{2}".format(lmbda, k, gamma)
	out_file = open(src_folder + "output/" + out_file_base + EXT, "w")


	# for each fold
	for fold_index in xrange(1, NUM_FOLDS + 1):

		print "*** \t FOLD {0} \t ***".format(fold_index)

		M_train = lil_matrix( (_N, _M) )
		M_test = lil_matrix( (_N, _M) )
		rmse = 0.0
		mae = 0.0
		# M_test = lil_matrix( (_N, _M) ) 	


		#################################################################################
		#
		#	Get training data and testing data
		#
		#################################################################################
		

		train_path = src_folder + base_file_name + TRAIN_PREFIX + str(fold_index) + EXT
		test_path = src_folder + base_file_name + TEST_PREFIX + str(fold_index) + EXT

		with open(train_path, "r") as infile:
			reader = csv.reader(infile, delimiter="\t")	
			for line in reader:
				userid = int(line[0], 10)
				movieid = int(line[1], 10)
				score = float(line[2])
				M_train[userid, movieid] = score


		with open(test_path, "r") as infile:

			reader = csv.reader(infile, delimiter="\t")	

			for line in reader:
				userid = int(line[0], 10)
				movieid = int(line[1], 10)
				score = float(line[2])
				M_test[userid, movieid] = score



		M_train = M_train.tocsr()
		M_test = M_test.tocsr()

		#################################################################################
		#
		#	Training phase
		#
		#################################################################################		
		print "Training phase.."

		pbar = ProgressBar(widgets=widgets, maxval=n_epochs).start()     
		count = 0


		# Index matrix for training data
		I_train = M_train.copy()
		for i in xrange(len(I_train.data)):
			I_train.data[i] = 1


		# Index matrix for test data
		I_test = M_test.copy()
		for i in xrange(len(I_test.data)):
			I_test.data[i] = 1


		# # Index matrix for training data
		# I_train = M_train.toarray().copy()
		# I_train[I_train > 0] = 1
		# I_train[I_train == 0] = 0

		# # Index matrix for test data
		# I_test = M_test.toarray().copy()
		# I_test[I_test > 0] = 1
		# I_test[I_test == 0] = 0



		P = 3 * np.random.rand(k, _N) # Latent user feature matrix initialization
		Q = 3 * np.random.rand(k, _M) # Latent movie feature matrix initialization

		users, items = M_train.nonzero() 
		train_errors_rmse = []
		train_errors_mae = []
		test_errors_rmse = []
		test_errors_mae = []

		
		# start gradient descent optimization
		for epoch in xrange(n_epochs):
			for u, i in zip(users, items):
				e = M_train[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient

				P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
				Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix


			if (fold_index == 1):	
				train_rmse, train_mae = compute_errors_light(M_train, I_train, P, Q) # Calculate rmse and mae error from train dataset	
				test_rmse, test_mae = compute_errors_light(M_test, I_test, P, Q)	# Calculate rmse and mae error from test dataset
				train_errors_rmse.append(train_rmse)
				train_errors_mae.append(train_mae)	
				test_errors_rmse.append(test_rmse)
				test_errors_mae.append(test_mae)



			count += 1
			pbar.update(count)


		pbar.finish()
		print "..done"

		if fold_index == 1:
			plot(n_epochs, train_errors_rmse, test_errors_rmse, train_errors_mae, test_errors_mae, True)

	


	
		#################################################################################
		#
		#	test phase
		#
		#################################################################################
		print "Test phase.."

		# write predictions only for first test (fold)
		# if (fold_index == 1):	
		# 	R_hat = prediction(P, Q)
		# 	rows, cols = M_test.nonzero()
		# 	for row, col in zip(rows,cols):
		# 		r_xi = R_hat[row, col]
		# 		out_file.write(str(row) + '\t' + str(col) + '\t' + str(r_xi) + '\n')


		# write predictions only for first test (fold)
		if fold_index == 1:
			users, items = M_test.nonzero()
			for u, i in zip(users, items):
				r_xi = prediction(P[:,u],Q[:,i])
				out_file.write(str(u) + '\t' + str(i) + '\t' + str(r_xi) + '\n')

		

		# update rmse and mae
		test_rmse, test_mae = compute_errors_light(M_test, I_test, P, Q)	# Calculate rmse and mae error from test dataset
		avg_rmse += test_rmse
		avg_mae += test_mae		

		print avg_rmse, avg_mae

		print "..done"
		print ""
				



	# average rmse and mae on validation folds
	eval_out_path = src_folder + "output/" + out_file_base + "_eval" + EXT

	with open(eval_out_path, "w") as file:
		file.write("RMSE" + "\t" + "MAE" + "\n")
		avg_rmse /= float(NUM_FOLDS)
		avg_mae /= float(NUM_FOLDS)
		file.write(str(avg_rmse) + "\t" + str(avg_mae))
	








def compute_errors(M, I, P, Q):

	R = prediction(P, Q)

	R[R < 1.0] = 1.0	# clamp in [1,5]
	R[R > 5.0] = 5.0	# clamp in [1,5]



	E = (I * (M.toarray() - R))

	rmse = np.sum( E**2 )
	rmse = np.sqrt(rmse / len(M.nonzero()[0]))

	mae = np.sum( np.absolute(E) )
	mae = np.sqrt(mae / len(M.nonzero()[0]))

	return rmse, mae





# for really big matrices
def compute_errors_light(M, I, P, Q):

	users, items = M.nonzero()
	rmse = 0.0
	mae = 0.0

	for u, i in zip(users, items):
		e = M[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
		mae += abs(e)
		rmse += e**2
		

	rmse = math.sqrt(rmse / float(len(users)))
	mae = math.sqrt(mae / float(len(users)))

	return rmse, mae



	


	



# Predict the unknown ratings through the dot product of the latent features for users and items 
def prediction(P, Q):
    return np.dot(P.T,Q)





def plot(n_epochs, train_errors_rmse, test_errors_rmse, train_errors_mae, test_errors_mae, show=False):
	plt.plot(range(n_epochs), train_errors_rmse, marker='o', label='Training Data');
	plt.plot(range(n_epochs), test_errors_rmse, marker='v', label='Test Data');
	plt.title('SGD-WR RMSE Learning Curve')
	plt.xlabel('Number of Epochs');
	plt.ylabel('RMSE');
	plt.legend()
	plt.grid()
	plt.savefig("sgd_rmse_curve_{0}-{1}-{2}.eps".format(lmbda, k, gamma), format="eps", dpi=1000, bbox_inches='tight')
	if show:
		plt.show()

	# clear
	plt.clf()

	plt.plot(range(n_epochs), train_errors_mae, marker='o', label='Training Data');
	plt.plot(range(n_epochs), test_errors_mae, marker='v', label='Test Data');
	plt.title('SGD-WR MAE Learning Curve')
	plt.xlabel('Number of Epochs');
	plt.ylabel('MAE');
	plt.legend()
	plt.grid()
	plt.savefig("sgd_mae_curve_{0}-{1}-{2}.eps".format(lmbda, k, gamma), format="eps", dpi=1000, bbox_inches='tight')
	if show:
		plt.show()



	
		
























###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")

	args = parser.parse_args()

	stochasticGradientDescentUV(args.filepath)
	




