#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           cf_item_item.py
#   author:         Ludovico Fabbri 1197400
#   description:    item-item collaborative filtering
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

# _N = 943 + 1
# _M = 1682 + 1

NUM_FOLDS = 5
_K = [2, 5, 10, 20]
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"
COSINE_SIMILARITY = "cos"
JACCARD_SIMILARITY = "jac"
PEARSON_CORR = "pears"
TF_IDF = "tfidf"
NONE = "none"
_P = 0.5





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





def build_tfidf_documents(src_folder):

	movie_tfidf_profiles = OrderedDict()


	# add folksonomy to profiles
	with io.open(src_folder + "folksonomy.csv", "r", encoding="ISO-8859-1") as file:
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10)
			tags = " ".join(tokens[1:])
			movie_tfidf_profiles[movieid] = tags


	# then add metadata to profiles
	with io.open(src_folder + "latest_small_metadata_full.csv", "r", encoding="ISO-8859-1") as file:
		file.readline()		# skip header
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[0], 10)
			tokens = tokens[3:]	
			text = " ".join(tokens)
			if movieid in movie_tfidf_profiles:
				movie_tfidf_profiles[movieid] += " " + text


			

	# # add folksonomy from small dataset
	# with io.open(src_folder + "tags.csv", "r", encoding="ISO-8859-1") as file:
	# 	file.readline()		# skip header
	# 	for line in file:
	# 		tokens = line.strip().split(",")
	# 		movieid = int(tokens[1], 10)
	# 		tag = tokens[2]
	# 		try:
	# 			movie_tfidf_profiles[movieid] += " " + tag
	# 		except Exception as e:
	# 			print line
	# 			print e

	# # add folksonomy from big dataset
	# with io.open("datasets/ml-20m/tags.csv", "r", encoding="ISO-8859-1") as file:
	# 	file.readline()		# skip header
	# 	for line in file:
	# 		tokens = line.strip().split(",")
	# 		movieid = int(tokens[1], 10)
	# 		tag = tokens[2]
	# 		try:
	# 			if movieid in movie_tfidf_profiles:
	# 				movie_tfidf_profiles[movieid] += " " + tag
	# 		except Exception as e:
	# 			print line
	# 			print e

	




	# private function used by TfidfVectorizer		
	def normalize(text):
		tokens = nltk.word_tokenize(text.lower().translate(remove_punctuation_map))
		stems = [stemmer.stem(t) for t in tokens]
		return stems	
	

	stemmer = PorterStemmer()
	
	remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

	vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

	# create an array of _M empty strings
	documents = [""] * _M


	for movieid in movie_tfidf_profiles:
		documents[movieid] = movie_tfidf_profiles[movieid]

	
	# print documents[2571]
	# exit()

	
	# doc1 = movie_tfidf_profiles[1270]
	# doc2 = movie_tfidf_profiles[2011]
	# print doc1
	# print doc2

	M_tfidf = vectorizer.fit_transform(documents)
	# tfidf = vectorizer.fit_transform([doc1, doc2])
	# print ((tfidf * tfidf.T).A)[1270,2011]
	
	return (M_tfidf * M_tfidf.T).A
	
		






def collaborativeFilteringItemItem(filepath, similarity, hybrid, tfidf):

	print _P

	src_folder = parseOutputFolderPath(filepath)
	base_file_name = parseFileName(filepath)

	if hybrid:
		out_file_base = "{0}_pred_item_hyb_{1}_".format(base_file_name, _P) + similarity 

	elif tfidf:
		out_file_base = "{0}_pred_item_tfidf_{1}_".format(base_file_name, _P) + similarity 
	
	else:
		out_file_base = "{0}_pred_item_".format(base_file_name) + similarity 

	out_file = open(src_folder + "output/" + out_file_base + EXT, "w")

	avg_rmse_dict = defaultdict(float)
	avg_mae_dict = defaultdict(float)


	# used only when hybrid = True ---------------------------------------------
	if hybrid:
		M_profiles = spio.loadmat(src_folder + "output/movie_profiles.mat")["M"]
		dots_profiles = M_profiles.dot(M_profiles.transpose())
	else:
		M_profiles = False 		# placeholder
		dots_profiles = False 	# placeholder

	# --------------------------------------------------------------------------


	# used only when tfidf = True ---------------------------------------------	
	if tfidf:
		print "Computing tf-idf matrix.."
		dots_tfidf = build_tfidf_documents(src_folder)
		print "..done"
	else:
		# just a placeholder
		dots_tfidf = False
	# --------------------------------------------------------------------------
	


	# for each fold
	for fold_index in xrange(1, NUM_FOLDS + 1):
		
		M = lil_matrix( (_N, _M) )
		M_NORM = lil_matrix( (_N, _M) )
		avg_users = defaultdict(float)
		avg_movies = defaultdict(float)

		rmse_dict = defaultdict(float)
		mae_dict = defaultdict(float)
		mu = 0.0
		similarities = defaultdict(float)



		print "*** \t FOLD {0} \t ***".format(fold_index)

		#################################################################################
		#
		#	training phase
		#
		#################################################################################
		print "Training Phase.."
		train_path = src_folder + base_file_name + TRAIN_PREFIX + str(fold_index) + EXT


		with open(train_path, "r") as file:

			count = 0
			count_users = defaultdict(int)
			count_movies = defaultdict(int)

			for line in file:

				tokens = line.strip().split("\t")

				userid = int(tokens[0], 10)
				movieid = int(tokens[1], 10)
				score = float(tokens[2])

				M[userid, movieid] = score
				M_NORM[userid, movieid] = score
				avg_users[userid] += score
				avg_movies[movieid] += score
				count_users[userid] += 1
				count_movies[movieid] += 1
				mu += score
				count += 1




		mu /= count

		for i in avg_users:
			avg_users[i] /= float(count_users[i])


		for j in avg_movies:
			avg_movies[j] /= float(count_movies[j])




		
		# Normalize utility matrix if Pearson Correlation similarity is selected	
		if similarity == PEARSON_CORR:
			np.seterr(divide='ignore', invalid='ignore')
			M_NORM = M_NORM.tocsc()	
			M_NORM = M_NORM.transpose()
			sums = np.array(M_NORM.sum(axis=1).squeeze())[0]
			counts = np.diff(M_NORM.indptr)
			means = np.nan_to_num(sums / counts)
			M_NORM.data -= np.repeat(means, counts)
			M_NORM = M_NORM.transpose()
			# M_NORM = M_NORM.tolil()
		


		# pbar = ProgressBar(widgets=widgets, maxval=_M).start()


		# if similarity == PEARSON_CORR:
		# 	# normalize matrix
		# 	for j in xrange(_M):
		# 		for i in M.getcol(j).nonzero()[0]:
		# 			M_NORM[i, j] = M_NORM[i, j] - avg_movies[j]
		# 		pbar.update(j)


		# pbar.finish()

		print "..done"



		#################################################################################
		#
		#	test phase
		#
		#################################################################################
		
		print "Test Phase.."
		test_path = src_folder + base_file_name + TEST_PREFIX + str(fold_index) + EXT
		M = M.tocsc()

		# print test_path

		
		if similarity != JACCARD_SIMILARITY:
			dots = M_NORM.transpose().dot(M_NORM)
		else:
			dots = False 	# placeholder


		item_vectors = defaultdict()	

		# used only for jaccard
		if similarity == JACCARD_SIMILARITY:	
			for j in xrange(_M):
				item_vectors[j] = set(M.getcol(j).nonzero()[0])	
		



		pbar = ProgressBar(widgets=widgets, maxval=21000).start()
		count = 0.0


		with open(test_path, "r") as file:

			for line in file:

				tokens = line.strip().split("\t")
				

				userid = int(tokens[0], 10)
				movieid = int(tokens[1], 10)
				score = float(tokens[2])
				count += 1
				pbar.update(count)




				if similarity == JACCARD_SIMILARITY:
					top_k = topK_JaccardSimilarity(M, similarities, userid, movieid, hybrid, item_vectors, M_profiles, tfidf, dots_tfidf) 
					
				else:
					top_k = topK_CosineSimilarity(M, dots, similarities, userid, movieid, hybrid, dots_profiles, tfidf, dots_tfidf)
					


				r_xi = 0.0

				for k in _K:
					num = 0.0
					den = 0.0
					b_xi = avg_users[userid] + avg_movies[movieid] - mu

					for t in top_k[:k]:
						curr_movieid = t[0]
						sim = t[1]
						b_xj = avg_users[userid] + avg_movies[curr_movieid] - mu

						num += sim * (M[userid, curr_movieid] - b_xj)
						den += sim

					if den != 0:
						r_xi = b_xi + (num / float(den))
					else:
						r_xi = b_xi


					# clamp prediction in [1,5]
					if r_xi < 1:
						r_xi = 1.0
					if r_xi > 5:
						r_xi = 5.0


					# update rmse and mae
					error = r_xi - score
					rmse_dict[k] += error**2
					mae_dict[k] += abs(error)

				# write predictions only for first test (fold)
				if (fold_index == 1):
					out_file.write(tokens[0] + '\t' + tokens[1] + '\t' + str(r_xi) + '\n')




		# normalize rmse and mae
		for k in _K:
			rmse_dict[k] = math.sqrt(rmse_dict[k] / float(count))
			mae_dict[k] = math.sqrt(mae_dict[k] / float(count))
			avg_rmse_dict[k] += rmse_dict[k]
			avg_mae_dict[k] += mae_dict[k]

		out_file.close()


		pbar.finish()
		print "..done\n"
		print ""
		







	# average rmse on number of folds for each neighbour size and write to disk
	eval_out_path = src_folder + "output/" + out_file_base + "_eval" + EXT

	with open(eval_out_path, "w") as file:

		file.write("k" + "\t" + "RMSE" + "\t" + "MAE" + "\n")

		for k in _K:
			avg_rmse_dict[k] /= 5.0
			avg_mae_dict[k] /= 5.0
			file.write(str(k) + "\t" + str(avg_rmse_dict[k]) + "\t" + str(avg_mae_dict[k]) + "\n")



		

	









def topK_CosineSimilarity(M, dots, similarities, userid, movieid, hybrid, dots_profiles, tfidf, dots_tfidf):

	top_k = []
	movieid_sq_norm = dots[movieid, movieid]

	# used only if hybrid = True
	if hybrid:
		movieid_prof_sq_norm = dots_profiles[movieid, movieid]




	for j in M.getrow(userid).nonzero()[1]:

		if (j == movieid):
			continue

		if (movieid, j) in similarities:
			sim = similarities[ (movieid, j) ]
			top_k.append( (j, sim) )
			continue

		if movieid_sq_norm == 0:
			similarities[ (movieid, j) ] = -1.0
			top_k.append( (j, -1.0) )
			continue

		j_sq_norm = dots[j, j]

		if j_sq_norm == 0:
			similarities[ (movieid, j) ] = -1.0
			top_k.append( (j, -1.0) )
			continue

		sim = dots[movieid, j] / math.sqrt(movieid_sq_norm * j_sq_norm)


		# if hybrid mode is selected, we compute linear combination between cf similarity and content-based similarity
		if hybrid:
			cb_sim = 0.0
			num = dots_profiles[movieid, j]
			if num != 0:
				den = math.sqrt(movieid_prof_sq_norm * dots_profiles[j, j])
				cb_sim = num / den

			sim = _P * cb_sim + (1 - _P) * sim
		# --------------------------------------------------------------------

		elif tfidf:
			cb_sim = dots_tfidf[movieid, j].item()
			sim = _P * cb_sim + (1 - _P) * sim

		# --------------------------------------------------------------------


		similarities[ (movieid, j) ] = sim
		similarities[ (j, movieid) ] = sim
		top_k.append( (j, sim) )


	top_k = sorted(top_k, key=lambda x: x[1], reverse=True)

	return top_k











def topK_JaccardSimilarity(M, similarities, userid, movieid, hybrid, item_vectors, M_profiles, tfidf, dots_tfidf):
	
	top_k = []
	sim = 0

	# extract feature set-vector from profile (rows are movies, columns are features) --> used only if hybrid = True
	if hybrid:
		movieid_prof_vect = set(M_profiles.getrow(movieid).nonzero()[1])

	# extract movieid vector from utility matrix
	# movieid_vect = set(M.getcol(movieid).nonzero()[0])

	# Jaccard similarity
	for j in M.getrow(userid).nonzero()[1]:

		if (j == movieid):
			continue

		if (movieid, j) in similarities:
			sim = similarities[ (movieid, j) ]
			continue

		elif (j, movieid) in similarities:
			sim = similarities[ (j, movieid) ]
			continue

		else:			
			union = len(item_vectors[movieid].union(item_vectors[j]))
			intersection = len(item_vectors[movieid].intersection(item_vectors[j]))
			sim = intersection / float(union)

			# j_vector = set(M.getcol(j).nonzero()[0])
			# union = len(movieid_vect.union(j_vector))
			# intersection = len(movieid_vect.intersection(j_vector))
			# sim = intersection / float(union)


		# if hybrid mode is selected, we compute linear combination between cf similarity and content-based similarity	
		if hybrid:
			cb_sim = 0.0
			j_v = set(M_profiles.getrow(j).nonzero()[1])
			num = len(movie_prof_vect.intersection(j_v))
			den = len(movie_prof_vect.union(j_v))
			cb_sim = num / float(den)
			sim = _P * cb_sim + (1 - _P) * sim
		# --------------------------------------------------------------------

		elif tfidf:
			cb_sim = dots_tfidf[movieid, j].item()
			sim = _P * cb_sim + (1 - _P) * sim

		# --------------------------------------------------------------------

		similarities[ (movieid, j) ] = sim
		similarities[ (j, movieid) ] = sim
		top_k.append((j, sim))

		

	# top items similar to movieid rated by userid
	top_k = sorted(top_k, key=lambda x: x[1], reverse=True)

	return top_k











###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")
	parser.add_argument("--sim", type=str, nargs="?", const=1, default=PEARSON_CORR, choices=[COSINE_SIMILARITY, JACCARD_SIMILARITY, PEARSON_CORR], help="similarity measure (string)")
	parser.add_argument("--hybrid", type=bool, nargs="?", const=1, default=False, help="Mix collaborative-filtering and feature-based similarities")
	parser.add_argument("--tfidf", type=bool, nargs="?", const=1, default=False, help="Mix collaborative-filtering and document-based similarities")
	parser.add_argument("--p", type=float, help="Linear combination factor")


	args = parser.parse_args()

	_P = args.p if args.p else _P

	if args.tfidf:
		args.hybrid = False


	collaborativeFilteringItemItem(args.filepath, args.sim, args.hybrid, args.tfidf)










