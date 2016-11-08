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
# import csv
# import unicodecsv as csv
import operator
import argparse
import numpy as np
import scipy as sp
from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict 
from collections import OrderedDict



OUT_FOLDER = "datasets/ml-latest-small/output/"
SRC_FOLDER = "datasets/ml-latest-small/"
_N = 671 + 1
_M = 164979 + 1



def getTaggedMovies():
	tagged_movies = set()
	with io.open("datasets/ml-latest-small/folksonomy.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10)
			tagged_movies.add(movieid)
	return tagged_movies




def build_unique_sorted_files():

	tagged_movies = getTaggedMovies()	# movies with at least 4 tags

	unique_years = set()
	unique_directors = set()
	unique_actors = set()
	unique_genres = []
	unique_tags = set()	


	with io.open(SRC_FOLDER + "ml-latest-small_metadata.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:

			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10)

			# skip all (< 4 tags) movies
			if movieid in tagged_movies:
				unique_years.add(tokens[6])
				directors = set([x.strip() for x in tokens[4].split(",")])
				actors = set([x.strip() for x in tokens[5].split(",")])
				unique_directors = unique_directors.union(directors)
				unique_actors = unique_actors.union(actors)
		


		unique_years = sorted(unique_years)
		unique_directors = sorted(unique_directors)
		unique_actors = sorted(unique_actors)
		



		with io.open(OUT_FOLDER + "unique_years.csv", "w", encoding="ISO-8859-1") as file:
			for x in unique_years:
				file.write(x + "\n")


		with io.open(OUT_FOLDER + "unique_directors.csv", "w", encoding="ISO-8859-1") as file:
			for x in unique_directors:
				file.write(x + "\n")


		with io.open(OUT_FOLDER + "unique_actors.csv", "w", encoding="ISO-8859-1") as file:
			for x in unique_actors:
				file.write(x + "\n")
				

		

	with io.open(SRC_FOLDER + "unique_genres.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			unique_genres.append(line.strip())




	with io.open(SRC_FOLDER + "folksonomy.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			tags = line.strip().split("\t")[1:]
			for tag in tags:
				unique_tags.add(tag)

		unique_tags = sorted(unique_tags)

		with io.open(OUT_FOLDER + "unique_tags.csv", "w", encoding="ISO-8859-1") as file:
			for tag in unique_tags:
				file.write(tag + "\n")




	with io.open(OUT_FOLDER + "all_features.csv", "w", encoding="ISO-8859-1") as file:

		for x in unique_years:
			file.write(x + "\n")
		
		for x in unique_directors:
			file.write(x + "\n")

		for x in unique_actors:
			file.write(x + "\n")

		for x in unique_genres:
			file.write(x + "\n")

		for x in unique_tags:
			file.write(x + "\n")








def buildMoviesProfiles():

	tagged_movies = getTaggedMovies()

	# there are some duplicates, thus we use a dictionary for each feature
	years_dict = OrderedDict()
	directors_dict = OrderedDict()
	actors_dict = OrderedDict()
	genres_dict = OrderedDict()
	tags_dict = OrderedDict()
	index = 1

	with io.open(OUT_FOLDER + "unique_years.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			years_dict[line.strip()] = index
			index += 1

	with io.open(OUT_FOLDER + "unique_directors.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			directors_dict[line.strip()] = index
			index += 1
			
	with io.open(OUT_FOLDER + "unique_actors.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			actors_dict[line.strip()] = index
			index += 1		

	with io.open(OUT_FOLDER + "unique_genres.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			genres_dict[line.strip()] = index
			index += 1

	with io.open(OUT_FOLDER + "unique_tags.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			tags_dict[line.strip()] = index
			index += 1



	num_features = len(years_dict) + len(directors_dict) + len(actors_dict) + len(genres_dict) + len(tags_dict)	
	M_movies = lil_matrix( (_M, num_features + 1) )		# + 1 to match index with the all_features.csv file		
	print M_movies._shape
	


	
	with io.open(SRC_FOLDER + "ml-latest-small_metadata.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10)
			year = tokens[3]
			directors = [x.strip() for x in tokens[4].split(",")]
			actors = [x.strip() for x in tokens[5].split(",")]

			M_movies[movieid, years_dict[year]] = 1.0

			for x in directors:
				M_movies[movieid, directors_dict[x]] = 1.0

			for x in actors:
				M_movies[movieid, actors_dict[x]] = 1.0

			


	with io.open(SRC_FOLDER + "movies.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:

			tokens = line.strip().split(",")
			movieid = int(tokens[0], 10)
			genres = [x.strip() for x in tokens[len(tokens) - 1].split("|")]

			for x in genres:
				try:
					M_movies[movieid, genres_dict[x]] = 1.0
				except:
					print "Genre KeyError: " + x



		
	


	with io.open(SRC_FOLDER + "folksonomy.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			tokens = line.strip().split("\t")
			movieid= int(tokens[0], 10)
			tags = tokens[1:]
			for tag in tags:
				M_movies[movieid, tags_dict[tag]] = 1.0

			

	
	spio.savemat(OUT_FOLDER + "movie_profiles", {"M" : M_movies})
			
		








def buildUserProfiles():

	M_ = lil_matrix( (_N, _M) )
	M_NORM = lil_matrix( (_N, _M) )
	avg_users = defaultdict(float)
	avg_movies = defaultdict(float)
	count_users = defaultdict(int)
	count_movies = defaultdict(int)


	with open(SRC_FOLDER + "ratings.csv", "r") as file:
		for line in file:

			tokens = line.strip().split(",")

			userid = int(tokens[0], 10) 
			movieid = int(tokens[1], 10) 
			score = float(tokens[2])

			M_[userid, movieid] = score
			avg_users[userid] += score
			avg_movies[movieid] += score
			count_users[userid] += 1
			count_movies[movieid] += 1
			


	for i in avg_users:
		avg_users[i] /= float(count_users[i])		

	for i in avg_movies:
		avg_movies[i] /= float(count_movies[i])



	# normalize matrix (by user average --> from book)
	for i in xrange(_N):
		for j in M_.getrow(i).nonzero()[1]:
			# M_NORM[i, j] = M_[i, j] - avg_movies[j]
			# M_NORM[i, j] = M_[i, j] - avg_users[i]
			M_NORM[i, j] = M_[i, j]



	M_movies = spio.loadmat(OUT_FOLDER + "movie_profiles")["M"].tolil()
	M_users = lil_matrix((_N, M_movies._shape[1]))
	
	print M_movies._shape
	print M_users._shape

	# build profiles
	for i in xrange(_N):

		count = 0

		for j in M_NORM.getrow(i).nonzero()[1]:

			for k in M_movies.getrow(j).nonzero()[1]:
				M_users[i, k] += M_NORM[i, j]

			count += 1

		for j in M_users.getrow(i).nonzero()[1]:
			M_users[i, j] /= float(count)


		



	# write to disk
	spio.savemat(OUT_FOLDER + "users_profiles", {"M" : M_users})
	  

		



















###############################################
#  main
###############################################
if __name__ == '__main__':



	build_unique_sorted_files()

	# buildMoviesProfiles()

	# buildUserProfiles()







