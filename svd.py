
#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           svd.py
#   author:         Ludovico Fabbri 1197400
#   description:    recommendations using Singular Value Decomposition
#
#
#################################################################################

import csv
import io
import argparse
import math
import numpy as np
import scipy.sparse as sp

from recsys.algorithm.factorize import SVD
from recsys.evaluation.prediction import MAE
from recsys.evaluation.prediction import RMSE


from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections import OrderedDict
import matplotlib.pyplot as plt


from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
	FileTransferSpeed, FormatLabel, Percentage, \
	ProgressBar, ReverseBar, RotatingMarker, \
	SimpleProgress, Timer



_N = 671 + 1
_M = 164979 + 1
_K = 20  # Dimension of the latent feature space
NUM_FOLDS = 5
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"
MIN_RATING = 1.0
MAX_RATING = 5.0




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






def svd(filepath):

	src_folder = parseOutputFolderPath(filepath)
	base_file_name = parseFileName(filepath)

	avg_rmse = 0.0
	avg_mae = 0.0

	out_file_base = base_file_name + "_pred_svd"
	out_file = open(src_folder + "output/" + out_file_base + EXT, "w")


	# for each fold
	for fold_index in xrange(1, NUM_FOLDS + 1):

		print "*** \t FOLD {0} \t ***".format(fold_index)

		M_test = lil_matrix( (_N, _M) )
		rmse = 0.0
		mae = 0.0

		train_path = src_folder + base_file_name + TRAIN_PREFIX + str(fold_index) + EXT
		test_path = src_folder + base_file_name + TEST_PREFIX + str(fold_index) + EXT

		print train_path
		print test_path



		svd = SVD()
		svd.load_data(filename=train_path,
		            sep=',',
		            format={'col':0, 'row':1, 'value':2, 'ids': float})

		svd.compute(k=_K,
		            min_values=1,
		            pre_normalize=None,
		            mean_center=True,
		            post_normalize=True)



		with open(test_path, "r") as infile:
			reader = csv.reader(infile, delimiter=",")	
			for line in reader:
				userid = int(line[0], 10)
				movieid = int(line[1], 10)
				score = float(line[2])
				M_test[userid, movieid] = score
				



		# GROUND_TRUTH = [3.0, 1.0, 5.0, 2.0, 3.0]
		# TEST = [2.3, 0.9, 4.9, 0.9, 1.5]
		# mae = MAE()
		# mae.load_ground_truth(GROUND_TRUTH)
		# mae.load_test(TEST)
		# mae.compute() #returns 0.7		




		

		# write predictions only for first test (fold)
		if (fold_index == 1):	
			rows, cols = M_test.nonzero()
			for row, col in zip(rows,cols):
				try:
					r_xi = svd.predict(col, row, MIN_RATING, MAX_RATING)
				except:
					print row, col
				out_file.write(str(row) + '\t' + str(col) + '\t' + str(r_xi) + '\n')

		


		print "..done"
		print ""

		exit()



	
	out_file.close()			


	# average rmse and mae on validation folds
	eval_out_path = src_folder + "output/" + out_file_base + "_eval" + EXT

	with open(eval_out_path, "w") as file:
		file.write("RMSE" + "\t" + "MAE" + "\n")
		avg_rmse /= float(NUM_FOLDS)
		avg_mae /= float(NUM_FOLDS)
		file.write(str(avg_rmse) + "\t" + str(avg_mae))	

	




def compute_errors(M, I, R_hat):
	# R_hat[R_hat < 1.0] = 1.0	# clamp in [1,5]
	# R_hat[R_hat > 5.0] = 5.0	# clamp in [1,5]

	E = (I * (M.toarray() - R_hat))

	rmse = np.sum( E**2 )
	rmse = np.sqrt(rmse / len(M.nonzero()[0]))

	mae = np.sum( np.absolute(E) )
	mae = np.sqrt(mae / len(M.nonzero()[0]))
	
	return rmse, mae









###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")

	args = parser.parse_args()

	svd(args.filepath)

	






