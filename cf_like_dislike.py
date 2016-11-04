#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           cf_like_dislike.py
#   author:         Ludovico Fabbri 1197400
#   description:    binary collaborative filtering based on jaccard similarity (like-dislike)
#
#
#################################################################################


import io
import argparse
import math
import numpy as np
from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections import OrderedDict

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

# progress bar settings
widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]



SRC_FOLDER = "datasets/ml-latest-small-binary/"
_N = 671
_M = 164979



def likeDislikeCollaborativeFiltering():


	M = lil_matrix((_N, _M))
	similarities = defaultdict()

	with open(SRC_FOLDER + "ratings_binary.csv") as file:

		for line in file:
			tokens = line.strip().split("\t")
			userid = int(tokens[0], 10) - 1
			movieid = int(tokens[1], 10) - 1
			score = int(tokens[2], 10)

			M[userid, movieid] = score


	out_file = open(SRC_FOLDER + "test.csv", "w")	

	pbar = ProgressBar(widgets=widgets, maxval=20010).start()
	row = 0	

	with open(SRC_FOLDER + "ratings_binary_test_1.csv", "r") as file:

		for line in file:
			tokens = line.strip().split("\t")
			userid = int(tokens[0], 10) - 1
			movieid = int(tokens[1], 10) - 1
			score = int(tokens[2], 10)
			row += 1
			pbar.update(row)



			pred = 0.0
			count = 0

			for i in M.getcol(movieid).nonzero()[0]:

				if i == userid:
					continue

				if M[i, movieid] == 1:
					pred += similarity(M, userid, i)
				else:
					pred -= similarity(M, userid, i)

				count += 1

				

			try:		
				pred /= float(count)
			except:
				pred = 0.0

			out_file.write(str(pred) + "\t" + str(score) + "\n")


	pbar.finish()		
	out_file.close()




		








def similarity(M, u_i, u_j):

	u_i_likes = set()
	u_i_dislikes = set()
	u_j_likes = set()
	u_j_dislikes = set()

	for j in M.getrow(u_i).nonzero()[1]:
		if M[u_i, j] == 1:
			u_i_likes.add(j)
		else:
			u_i_dislikes.add(j)



	
	for j in M.getrow(u_j).nonzero()[1]:
		if M[u_j, j] == 1:
			u_j_likes.add(j)
		else:
			u_j_dislikes.add(j)


	# print u_i_likes
	# print u_i_dislikes
	# print u_j_likes
	# print u_j_dislikes


	num = len(u_i_likes.intersection(u_j_likes)) + len(u_i_dislikes.intersection(u_j_dislikes)) - len(u_i_likes.intersection(u_j_dislikes)) + len(u_i_dislikes.intersection(u_j_likes))
	den = len (u_i_likes.union(u_i_dislikes).union(u_j_likes).union(u_j_dislikes))

	sim = num / float(den)

	return sim		























###############################################
#  main
###############################################
if __name__ == '__main__':

    

    likeDislikeCollaborativeFiltering()







