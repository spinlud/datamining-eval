#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           build_sample.py
#   author:         Ludovico Fabbri 1197400
#   description:    build a smaller sample of the dataset (about 10% of the users)
#
#
#################################################################################


import argparse
from random import random
from collections import OrderedDict


global THRESHOLD
global TAG



def buildSample(threshold):

	users_dict = OrderedDict()

	with open("datasets/ml-latest-small/ratings.csv", "r") as file:

		for line in file:
			line = line.strip()
			tokens = line.split(",")
			userid = tokens[0]

			if userid in users_dict:
				users_dict[userid].append(line)
			else:
				users_dict[userid] = []
				users_dict[userid].append(line)

	count = 0

	with open("datasets/ml-latest-small/ratings_sample{0}.csv".format(TAG), "w") as file:

		for userid in users_dict:

			rand = random()

			if rand <= threshold:
				count += 1
				for line in users_dict[userid]:
					file.write(line + "\n")



	print "Original # unique users:\t" +  str(len(users_dict))
	print "Sample   # unique users:\t" +  str(count)







def buildSparseSample(threshold):

	with open("datasets/ml-latest-small/ratings.csv", "r") as infile:

		with open("datasets/ml-latest-small/ratings_sample_sparse{0}.csv".format(TAG), "w") as outfile:

			for line in infile:
				rand = random()
				if rand <= threshold:
					outfile.write(line.strip() + "\n")

					



	
	
	



###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--th", type=float, choices=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9], help="percentage of users to keep in the test data")	
	parser.add_argument("--tag", type=str, help="file output tag")	

	args = parser.parse_args()

	global TAG
	TAG = "_{0}".format(args.tag) if args.tag else ""

	global THRESHOLD
	THRESHOLD = args.th if args.th else 0.2

	# buildSample(THRESHOLD)
	buildSparseSample(THRESHOLD)

