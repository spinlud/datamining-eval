#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           knn_class.py
#   author:         Ludovico Fabbri 1197400
#   description:    classification using k-Nearest Neighbors
#
#
#################################################################################


import io
import time
import csv
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

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
	FileTransferSpeed, FormatLabel, Percentage, \
	ProgressBar, ReverseBar, RotatingMarker, \
	SimpleProgress, Timer

# progress bar settings
widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]


_N = 671 + 1
_M = 164979 + 1


NUM_FOLDS = 5
_K = [2, 5, 10, 20]
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"
COSINE_DIST = "cos"
EUCLIDEAN_DIST = "euc"

TF_IDF = "tfidf"
NONE = "none"
_P = 0.25





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







def kNN(filepath, distance, hybrid, tfidf):

	src_folder = parseOutputFolderPath(filepath)
	base_file_name = parseFileName(filepath)

	out_file_base = "{0}_class_knn_".format(base_file_name) + distance 
	out_file = open(src_folder + "output/" + out_file_base + EXT, "w")


	avg_precision = 0.0
	avg_recall = 0.0
	avg_f1 = 0.0




	# for each fold
	for fold_index in xrange(1, NUM_FOLDS + 1):

		print "*** \t FOLD {0} \t ***".format(fold_index)
		
		M = lil_matrix( (_N, _M) )
		true_positives = 0
		true_negatives = 0
		false_positives = 0
		false_negatives = 0
		distances = defaultdict(float)

		#################################################################################
		#
		#	training phase
		#
		#################################################################################
		print "Collecting train sample points.."
		train_path = src_folder + base_file_name + TRAIN_PREFIX + str(fold_index) + EXT


		with open(train_path, "r") as file:
			for line in file:
				tokens = line.strip().split("\t")
				userid = int(tokens[0], 10)
				movieid = int(tokens[1], 10)
				label = float(tokens[2])
				M[userid, movieid] = label


		print "..done"





		#################################################################################
		#
		#	test phase
		#
		#################################################################################

		print "Test phase.."		
		test_path = src_folder + base_file_name + TEST_PREFIX + str(fold_index) + EXT
		dots = M.transpose().dot(M)

		pbar = ProgressBar(widgets=widgets, maxval=20050).start()
		count = 0

		with open(test_path, "r") as file:
			
			for line in file:

				tokens = line.strip().split("\t")
				
				userid = int(tokens[0], 10)
				movieid = int(tokens[1], 10)
				label = int(tokens[2], 10)	# 1 = relevant, -1 = irrelevant
				count += 1
				pbar.update(count)


				top_k = []
				movieid_sq_norm = dots[movieid, movieid]

				# per ogni movie visto da userid
				for j in M.getrow(userid).nonzero()[1]:

					if (j == movieid):
						continue

					if (movieid, j) in distances:
						sim = distances[ (movieid, j) ]
						top_k.append( (j, sim) )
						continue

					if movieid_sq_norm == 0:
						distances[ (movieid, j) ] = -1.0
						top_k.append( (j, -1.0) )
						continue

					j_sq_norm = dots[j, j]

					if j_sq_norm == 0:
						distances[ (movieid, j) ] = -1.0
						top_k.append( (j, -1.0) )
						continue

					dist = 1 - dots[movieid, j] / math.sqrt(movieid_sq_norm * j_sq_norm)
					distances[ (movieid, j) ] = dist
					distances[ (j, movieid) ] = dist
					top_k.append( (j, dist) )


				top_k = sorted(top_k, key=lambda x: x[1]) [:10]

				


				likes = 0
				dislikes = 0	

				# count labels from top_k neighbors
				for t in top_k:	
					j_label = M[userid, t[0]]
					if j_label == 1.0:
						likes += 1
					else:
						dislikes += 1


				# classify using majority voting from neighbors
				assigned_class = 1 if likes > dislikes else -1


				# write classification results only for first test (fold)
				if fold_index == 1:
					out_file.write(tokens[0] + '\t' + tokens[1] + '\t' + str(assigned_class) + '\n')


				# update for precision and recall
				if label == 1:
					if assigned_class == 1:
						true_positives += 1
					else:
						false_negatives += 1

				else:
					if assigned_class == 1:
						false_positives += 1
					else:
						true_negatives += 1



		precision = Precision(true_positives, false_positives)
		recall = Recall(true_positives, false_negatives)
		f1 = F1_measure(precision, recall)

		avg_precision += precision
		avg_recall += recall
		avg_f1 += f1

		print precision, recall, f1

		out_file.close()
		pbar.finish()
		print "..done"
		print ""




	# average measures on number of folds				
	avg_precision /= float(NUM_FOLDS)
	avg_recall /= float(NUM_FOLDS)
	avg_f1 /= float(NUM_FOLDS)

	# write on disk
	eval_out_path = src_folder + "output/" + out_file_base + "_eval" + EXT

	with open(eval_out_path, "w") as file:
		file.write("PRECISION" + "\t" + "RECALL" + "\t" + "F1" + "\n")
		file.write(str(avg_precision) + "\t" + str(avg_recall) + "\t" + str(avg_f1) + "\n")





					










def Precision(true_positives, false_positives):
	return true_positives / float(true_positives + false_positives)



def Recall(true_positives, false_negatives):
	return true_positives / float(true_positives + false_negatives)


def F1_measure(precision, recall):
	return (2 * precision * recall) / float(precision + recall)







				








###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")
	parser.add_argument("--d", type=str, nargs="?", const=1, default=COSINE_DIST, choices=[COSINE_DIST, EUCLIDEAN_DIST], help="distance measure (string)")
	parser.add_argument("--hybrid", type=bool, nargs="?", const=1, default=False, help="Mix collaborative-filtering and feature-based similarities")
	parser.add_argument("--tfidf", type=bool, nargs="?", const=1, default=False, help="Mix collaborative-filtering and document-based similarities")
	parser.add_argument("--p", type=float, help="Linear combination factor")


	args = parser.parse_args()

	_P = args.p if args.p else _P

	if args.tfidf:
		args.hybrid = False


	kNN(args.filepath, args.d, args.hybrid, args.tfidf)


