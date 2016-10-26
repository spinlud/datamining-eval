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
N = 671
M = 164979






def build_unique_sorted_files():

	unique_directors = set()
	unique_actors = set()
	unique_genres = []
	unique_tags = set()	


	with io.open(SRC_FOLDER + "ml-latest-small_metadata.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:

			tokens = line.strip().split("\t")
			directors = set([x.strip() for x in tokens[5].split(",")])
			actors = set([x.strip() for x in tokens[6].split(",")])
		


			unique_directors = unique_directors.union(directors)
			unique_actors = unique_actors.union(actors)
		


		unique_directors = sorted(unique_directors)
		unique_actors = sorted(unique_actors)
		





		with io.open(OUT_FOLDER + "unique_directors.csv", "w", encoding="ISO-8859-1") as file:
			for x in unique_directors:
				file.write(x + "\n")


		with io.open(OUT_FOLDER + "unique_actors.csv", "w", encoding="ISO-8859-1") as file:
			for x in unique_actors:
				file.write(x + "\n")
				

		

	with io.open(SRC_FOLDER + "unique_genres.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			unique_genres.append(line.strip())





	with io.open(SRC_FOLDER + "tags.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:
			tokens = line.strip().split(",")
			tag = tokens[2]
			unique_tags.add(tag)


		unique_tags = sorted(unique_tags)


		with io.open(OUT_FOLDER + "unique_tags.csv", "w", encoding="ISO-8859-1") as file:
			for tag in unique_tags:
				file.write(tag + "\n")




	with io.open(OUT_FOLDER + "all_features.csv", "w", encoding="ISO-8859-1") as file:
		
		for x in unique_directors:
			file.write(x + "\n")

		for x in unique_actors:
			file.write(x + "\n")

		for x in unique_genres:
			file.write(x + "\n")

		for x in unique_tags:
			file.write(x + "\n")








def buildMoviesProfiles():

	# there are some duplicates, thus we use a dictionary for each feature
	directors_dict = OrderedDict()
	actors_dict = OrderedDict()
	genres_dict = OrderedDict()
	tags_dict = OrderedDict()
	index = 0

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



	num_features = len(directors_dict) + len(actors_dict) + len(genres_dict) + len(tags_dict)	
	M_movies = lil_matrix( (M, num_features) )		
	print M_movies._shape
	


	
	with io.open(SRC_FOLDER + "ml-latest-small_metadata.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10) - 1
			directors = [x.strip() for x in tokens[5].split(",")]
			actors = [x.strip() for x in tokens[6].split(",")]


			for x in directors:
				M_movies[movieid, directors_dict[x]] = 1

			for x in actors:
				M_movies[movieid, actors_dict[x]] = 1
				


	with io.open(SRC_FOLDER + "movies.csv", "r", encoding="ISO-8859-1") as file:

		file.readline()		# skip header

		for line in file:

			tokens = line.strip().split(",")
			movieid = int(tokens[0], 10) - 1
			genres = [x.strip() for x in tokens[len(tokens) - 1].split("|")]

			for x in genres:
				try:
					M_movies[movieid, genres_dict[x]] = 1
				except:
					print "Genre KeyError: " + x


	


	with io.open(SRC_FOLDER + "tags.csv", "r", encoding="ISO-8859-1") as file:
	
		file.readline()		# skip header

		for line in file:
			tokens = line.strip().split(",")
			movieid= int(tokens[1], 10) - 1
			tag = tokens[2]

			M_movies[movieid, tags_dict[tag]] = 1



	
	spio.savemat(OUT_FOLDER + "movie_profiles", {"M" : M_movies})
			
		







def buildUserProfiles():

	M_ = lil_matrix( (N, M) )
	M_NORM = lil_matrix( (N, M) )
	avg_movies = defaultdict(float)
	count_movies = defaultdict(int)


	with open(SRC_FOLDER + "ratings.csv", "r") as file:
		for line in file:

			tokens = line.strip().split(",")

			userid = int(tokens[0], 10) - 1
			movieid = int(tokens[1], 10) - 1
			score = float(tokens[2])

			M_[userid, movieid] = score
			
			avg_movies[movieid] += score
			
			count_movies[movieid] += 1
			


	for i in avg_movies:
		avg_movies[i] /= float(count_movies[i])



	# normalize matrix
	for i in xrange(N):
		for j in M_.getrow(i).nonzero()[1]:
			M_NORM[i, j] = M_[i, j] - avg_movies[j]



	M_movies = spio.loadmat(OUT_FOLDER + "movie_profiles")["M"].tolil()
	M_users = lil_matrix((N, M_movies._shape[1]))
	
	print M_movies._shape
	print M_users._shape

	# build profiles
	for i in xrange(N):

		count = 0

		for j in M_NORM.getrow(i).nonzero()[1]:

			print j

			for k in M_movies.getrow(j).nonzero()[1]:
				# M_NORM[i, j] * M_movies.getrow(j)
				M_users[i, k] += M_NORM[i, j] * M_movies[j, k]

			count += 1

		for j in M_users.getrow(i).nonzero()[1]:
			M_users[i, j] /= float(count)



	# write to disk
	spio.savemat(OUT_FOLDER + "users_profiles", {"M" : M_users})
	  

		



















###############################################
#  main
###############################################
if __name__ == '__main__':

	# parser = argparse.ArgumentParser()

	# parser.add_argument("metadata_filepath", type=str, help="metadata file source path (string)")
	# parser.add_argument("tags_filepath", type=str, help="tags file source path (string)")
	

	# args = parser.parse_args()

	# build_unique_sorted_files(args.metadata_filepath, args.tags_filepath)



	# build_unique_sorted_files()

	# buildMoviesProfiles()

	buildUserProfiles()







