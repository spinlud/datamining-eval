#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           cb-profiles.py
#   author:         Ludovico Fabbri 1197400
#   description:    build items and user profiles based on metadata (directors, actors, genres) and tags
#
#
#################################################################################


import io
import math
# import csv
# import unicodecsv as csv
import argparse
import numpy as np
import scipy as sp
from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections import OrderedDict

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
	FileTransferSpeed, FormatLabel, Percentage, \
	ProgressBar, ReverseBar, RotatingMarker, \
	SimpleProgress, Timer

# progress bar settings
widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]




OUT_FOLDER = "datasets/ml-latest-small/output/"
SRC_FOLDER = "datasets/ml-latest-small/"
FILE_NAME_BASE = "ratings"
N = 671
M = 164979
NUM_FOLDS = 5
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"





def cbRecommender():

	M_movies = spio.loadmat(OUT_FOLDER + "movie_profiles")["M"]
	out_predictions_path = OUT_FOLDER + "cb_predictions.csv"
	out_pred_file = open(out_predictions_path, "w")
	out_pred_file_norm = open(OUT_FOLDER + "cb_predictions_norm.csv", "w")


	# for each fold 
	for fold_index in xrange(1, NUM_FOLDS + 1):

		_M = lil_matrix( (N, M) )
		M_NORM = lil_matrix( (N, M) )
		M_users = lil_matrix((N, M_movies._shape[1]))
		rmse_dict = defaultdict(float)
		avg_users = defaultdict(float)
		avg_movies = defaultdict(float)

		print "*** \t FOLD {0} \t ***".format(fold_index)

		#################################################################################
		#
		#	training phase
		#
		#################################################################################
		print "Training Phase.."
		train_path = SRC_FOLDER + FILE_NAME_BASE + TRAIN_PREFIX + str(fold_index) + EXT

		with open(train_path, "r") as file:

			count_users = defaultdict(int)
			count_movies = defaultdict(int)

			for line in file:
				tokens = line.strip().split(",")
				userid = int(tokens[0], 10) - 1
				movieid = int(tokens[1], 10) - 1
				score = float(tokens[2])

				_M[userid, movieid] = score
				avg_users[userid] += score
				avg_movies[movieid] += score
				count_users[userid] += 1
				count_movies[movieid] += 1


		for x in avg_users:
			avg_users[x] /= float(count_users[x])

		for x in avg_movies:
			avg_movies[x] /= float(count_movies[x])


		for i in xrange (N):
			for j in _M.getrow(i).nonzero()[1]:
				# M_NORM[i, j] = _M[i, j] - avg_movies[j]
				# M_NORM[i, j] = _M[i, j] - avg_users[i]
				M_NORM[i, j] = _M[i, j]


		pbar = ProgressBar(widgets=widgets, maxval=N).start()

		# build users profiles
		for i in xrange(N):

			counts = defaultdict(int)

			for j in M_NORM.getrow(i).nonzero()[1]:

				for k in M_movies.getrow(j).nonzero()[1]:
					M_users[i, k] += M_NORM[i, j]
					counts[k] += 1

				

			for k in M_users.getrow(i).nonzero()[1]:
				M_users[i, k] /= float(counts[k])

			pbar.update(i)

			


		pbar.finish()






		#################################################################################
		#
		#	test phase
		#
		#################################################################################

		print "Test Phase.."
		test_path = SRC_FOLDER + FILE_NAME_BASE + TEST_PREFIX + str(fold_index) + EXT		
		count = 0

		pbar = ProgressBar(widgets=widgets, maxval=21000).start()

		with open(test_path, "r") as file:

			for line in file:
				tokens = line.strip().split(",")
				userid = int(tokens[0], 10) - 1
				movieid = int(tokens[1], 10) - 1
				score = float(tokens[2])
				count += 1
				pbar.update(count)


				num = M_users.getrow(userid).dot( M_movies.getrow(movieid).transpose() )[0, 0].item()

				u_norm = math.sqrt (M_users.getrow(userid).dot(M_users.getrow(userid).transpose())[0, 0].item())
				m_norm = math.sqrt (M_movies.getrow(movieid).dot(M_movies.getrow(movieid).transpose())[0, 0].item())
				den = u_norm * m_norm

				if (den == 0):
					prediction = 0.0
				else:
					prediction = num / float(den)

				prediction_normalized = normalizeInRangeLinear(prediction, 1, 5)

				error = score - prediction_normalized
				rmse_dict[fold_index] += error**2

				# write predictions for the first test
				if (fold_index == 1):
					out_pred_file.write(tokens[0] + "\t" + tokens[1] + "\t" + str(prediction) + "\n")
					out_pred_file_norm.write(tokens[0] + "\t" + tokens[1] + "\t" + str(prediction_normalized) + "\n")


		rmse_dict[fold_index] = math.sqrt(rmse_dict[fold_index] / float(count))


	avg_rmse = 0.0
	for key in rmse_dict:
		avg_rmse += rmse_dict[key]
	avg_rmse /= float(NUM_FOLDS)

	with open(OUT_FOLDER + "cb_rmse.csv", "w") as file:
		file.write(str(avg_rmse) + "\n")


	out_pred_file.close()
	out_pred_file_norm.close()


	pbar.finish()
	print ""








def recommender():

	M_m = spio.loadmat(OUT_FOLDER + "movie_profiles")["M"]
	M_u = spio.loadmat(OUT_FOLDER + "users_profiles")["M"]

	print M_u._shape
	print M_m._shape


	i = 18
	j = 349


	with open(SRC_FOLDER + "ratings_test_1.csv", "r") as file:

		file.readline()

		i = 1

		for line in file:
			tokens = line.strip().split(",")
			userid = int(tokens[0], 10) - 1
			movieid = int(tokens[1], 10) - 1
			score = float(tokens[2])


			u_norm = math.sqrt (M_u.getrow(userid).dot(M_u.getrow(userid).transpose())[0, 0].item())
			m_norm = math.sqrt (M_m.getrow(movieid).dot(M_m.getrow(movieid).transpose())[0, 0].item())


			num = M_u.getrow(userid).dot( M_m.getrow(movieid).transpose() )[0, 0].item()
			den = u_norm * m_norm

			estimate = num / den
			print i, "\t", normalizeInRangeLinear(estimate, 1, 5)

			if i == 20:
				exit()
			else:
				i += 1






# normalizeInRangeLinear
def normalizeInRangeLinear(value, min_range, max_range):

	min_value = 0
	max_value = 1

	range1 = max_value - min_value
	range2 = max_range - min_range

	# for i in xrange (len(data)):
	# 	data[i] = ((data[i] - min_value) / float(range1) * range2) + min_range


	result = ((value - min_value) / float(range1) * range2) + min_range

	return result













###############################################
#  main
###############################################
if __name__ == '__main__':


	cbRecommender()






