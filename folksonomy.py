#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           folksonomy.py
#   author:         Ludovico Fabbri 1197400
#   description:    build folksonomy (active users that have used at least 4 tags, movies with at least 4 tags)
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



U_THRS = 4
THRESHOLD = 4



def getDatasetMovies():
	movies = set()
	with open("datasets/ml-latest-small/movies.csv", "r") as file:
		file.readline()
		for line in file:
			tokens = line.strip().split(",")
			movieid = int(tokens[0], 10)
			movies.add(movieid)
	return movies


def getDatasetUsers():
	users = set()
	with open("datasets/ml-latest-small/total_ratings.csv", "r") as file:
		for line in file:
			tokens = line.strip().split("\t")
			userid = int(tokens[0], 10)
			users.add(userid)
	return users





def buildFolksonomy():

	users = getDatasetUsers()			# dataset users
	movies = getDatasetMovies()			# dataset movies
	tagged_movies = defaultdict()		# movies with at least 4 tags
	users_dict = defaultdict()			# users that have tagged at least 4 movies
	active_users_count = 0

	
	

	with io.open("datasets/ml-latest-small/tags.csv", "r", encoding="ISO-8859-1") as file:
		file.readline()		#skip header
		for line in file:
			tokens = line.strip().split(",")
			userid = int(tokens[0], 10)
			movieid = int(tokens[1], 10)
			tag = tokens[2]

			if movieid in tagged_movies:
				tagged_movies[movieid].add( (userid, tag) )
			else:
				tagged_movies[movieid] = set()
				tagged_movies[movieid].add( (userid, tag) )

			if userid in users_dict:
				users_dict[userid].add(tag)
			else:
				users_dict[userid] = set()
				users_dict[userid].add(tag)




	with io.open("datasets/ml-latest-small/tags_big.csv", "r", encoding="ISO-8859-1") as file:
		file.readline()		#skip header
		for line in file:
			tokens = line.strip().split(",")
			userid = int(tokens[0], 10)
			movieid = int(tokens[1], 10)
			tag = tokens[2]


			if movieid in movies:
				if movieid in tagged_movies:
					tagged_movies[movieid].add( (userid, tag) )
				else:
					tagged_movies[movieid] = set()
					tagged_movies[movieid].add( (userid, tag) )


			if userid in users:
				if userid in users_dict:
					users_dict[userid].add(tag)
				else:
					users_dict[userid] = set()
					users_dict[userid].add(tag)




	# write on disk active users (users that have used at least 4 tags)
	with open("datasets/ml-latest-small/active_users.csv", "w") as file:
		for userid in sorted(users_dict):
			if len(users_dict[userid]) >= U_THRS:
				file.write(str(userid) + "\n")
				active_users_count += 1
	# ------------------------------------------------------------------------





	# Build set of tags used at least 4 times (remove some noise)				
	tags_used_at_least = set()
	tags_counts = defaultdict(int)

	for movieid in tagged_movies:
		for t in tagged_movies[movieid]:
			tag = t[1]
			tags_counts[tag] += 1


	for tag in tags_counts:
		if tags_counts[tag] >= THRESHOLD:
			tags_used_at_least.add(tag)
	# ------------------------------------------------------------------------


	# update tagged_movies removing unused tags from above
	for movieid in tagged_movies:
		toremove = []
		for t in tagged_movies[movieid]:
			tag = t[1]
			if tag not in tags_used_at_least:
				toremove.append(t)
		for t in toremove:
			tagged_movies[movieid].remove(t)
	# ------------------------------------------------------------------------



	count = 0

	# write on disk
	with io.open("datasets/ml-latest-small/folksonomy.csv", "w", encoding="ISO-8859-1") as file:
		for movieid in sorted(tagged_movies):

			# we filter out movies with less than 4 tags (remove some noise)
			if len(tagged_movies[movieid]) < THRESHOLD:
				continue

			line = str(movieid) + "\t"
			for t in tagged_movies[movieid]:
				tag = t[1]
				line += tag + "\t"
			line = line.strip() + "\n"
			file.write(line)

			count += 1



	total = len(movies)
	filtered = total - count

	print "Total movies: {0}".format(total)
	print "Tagged movies with at least {0} tags: {1}".format(THRESHOLD, count)
	print "Filtered: {0}".format(filtered)
	print ""

	








###############################################
#  main
###############################################
if __name__ == '__main__':

	buildFolksonomy()









