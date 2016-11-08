#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           dataset_filter.py
#   author:         Ludovico Fabbri 1197400
#   description:    filter dataset keeping only tagged movies derived from folksonomy.py script
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



def getTaggedMovies():
	tagged_movies = set()
	with io.open("datasets/ml-latest-small/folksonomy.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10)
			tagged_movies.add(movieid)
	return tagged_movies


def getActiveUsers():
	active_users = set()
	with open("datasets/ml-latest-small/active_users.csv", "r") as file:
		for line in file:
			userid = int(line.strip(), 10)
			active_users.add(userid)
	return active_users



def filterDataset():

	active_users = getActiveUsers()
	tagged_movies = getTaggedMovies()

	total = 0
	count = 0
	filtered = 0

	with open("datasets/ml-latest-small/total_ratings.csv", "r") as infile:
		with open ("datasets/ml-latest-small/ratings.csv", "w") as outfile:
			for line in infile:
				line = line.strip()
				tokens = line.split("\t")
				userid = int(tokens[0], 10)
				movieid = int(tokens[1], 10)

				if userid in active_users and movieid in tagged_movies:
					outfile.write(line + "\n")
					count += 1

				total += 1


	filtered = total - count

	print "Total ratings: {0}".format(total)
	print "Final ratings: {0}".format(count)
	print "Filtered: {0}".format(filtered)





###############################################
#  main
###############################################
if __name__ == '__main__':

	filterDataset()









