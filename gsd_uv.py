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



_N = 671 + 1
_M = 164979 + 1
NUM_FOLDS = 5
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"

lmbda = 0.1 # Regularisation weight
k = 20  # Dimension of the latent feature space
n_epochs = 50  # Number of epochs
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

	out_file_base = base_file_name + "_pred_sgd_uv"
	out_file = open(src_folder + "output/" + out_file_base + EXT, "w")


	for fold_index in xrange(1, NUM_FOLDS + 1):

		print "*** \t FOLD {0} \t ***".format(fold_index)

		M_train = lil_matrix( (_N, _M) )
		M_test = lil_matrix( (_N, _M) )
		rmse = 0.0
		mae = 0.0
		# M_test = lil_matrix( (_N, _M) ) 	


		#################################################################################
		#
		#	training phase
		#
		#################################################################################
		print "Training phase.."

		train_path = src_folder + base_file_name + TRAIN_PREFIX + str(fold_index) + EXT

		with open(train_path, "r") as infile:
			reader = csv.reader(infile, delimiter=",")	
			for line in reader:
				userid = int(line[0], 10)
				movieid = int(line[1], 10)
				score = float(line[2])
				M_train[userid, movieid] = score

		pbar = ProgressBar(widgets=widgets, maxval=n_epochs).start()     
		count = 0

		# Index matrix for training data
		I = M_train.toarray().copy()
		I[I > 0] = 1
		I[I == 0] = 0

		P = 3 * np.random.rand(k, _N) # Latent user feature matrix initialization
		Q = 3 * np.random.rand(k, _M) # Latent movie feature matrix initialization

		users, items = M_train.nonzero() 
		train_errors = []
		test_erros = []

		# start gradient descent optimization
		for epoch in xrange(n_epochs):
			for u, i in zip(users, items):
				e = M_train[u, i] - prediction(P[:,u],Q[:,i])  # Calculate error for gradient
				P[:,u] += gamma * ( e * Q[:,i] - lmbda * P[:,u]) # Update latent user feature matrix
				Q[:,i] += gamma * ( e * P[:,u] - lmbda * Q[:,i])  # Update latent movie feature matrix

			train_rmse = compute_rmse(M_train, I, P, Q) # Calculate root mean squared error from train dataset	
			train_errors.append(train_rmse)	
			count += 1
			pbar.update(count)


		pbar.finish()

		# prediction matrix is the dot product between P.T and Q 
		R_hat = prediction(P, Q)

		print "..done"


		plot(n_epochs, train_errors)



		#################################################################################
		#
		#	test phase
		#
		#################################################################################
		print "Test phase.."
		test_path = src_folder + base_file_name + TEST_PREFIX + str(fold_index) + EXT

		with open(test_path, "r") as infile:

			reader = csv.reader(infile, delimiter=",")	
			count = 0

			for line in reader:
				userid = int(line[0], 10)
				movieid = int(line[1], 10)
				score = float(line[2])
				# M_test[userid, movieid] = score

				r_xi = R_hat[userid, movieid]

				# clamp in [1, 5]
				r_xi = 1.0 if r_xi < 1.0 else r_xi
				r_xi = 5.0 if r_xi > 5.0 else r_xi

				error = r_xi - score
				rmse += error**2
				mae += abs(error)
				count += 1

				# write predictions only for first test (fold)
				if (fold_index == 1):
					out_file.write(line[0] + '\t' + line[1] + '\t' + str(r_xi) + '\n')


		# normalize rmse and mae
		rmse = math.sqrt(rmse / float(count))
		mae = math.sqrt(mae / float(count))
		avg_rmse += rmse
		avg_mae += mae




		out_file.close()
		print ""	


	# average rmse and mae on validation folds
	eval_out_path = src_folder + "output/" + out_file_base + "_eval" + EXT

	with open(eval_out_path, "w") as file:
		file.write("RMSE" + "\t" + "MAE" + "\n")
		avg_rmse /= float(NUM_FOLDS)
		avg_mae /= float(NUM_FOLDS)
		file.write(str(avg_rmse) + "\t" + str(avg_mae))






def compute_rmse(M, I, P, Q):	
	rmse = np.sum( (I * (M.toarray() - prediction(P, Q)))**2 )
	rmse = np.sqrt(rmse / len(M.nonzero()[0]))
	print rmse
	return rmse


	


	



# Predict the unknown ratings through the dot product of the latent features for users and items 
def prediction(P, Q):
    return np.dot(P.T,Q)





def plot(n_epochs, train_errors):
	plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data');
	# plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data');
	plt.title('SGD-WR Learning Curve')
	plt.xlabel('Number of Epochs');
	plt.ylabel('RMSE');
	plt.legend()
	plt.grid()
	plt.show()



	
		
























###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")

	args = parser.parse_args()

	stochasticGradientDescentUV(args.filepath)
	




