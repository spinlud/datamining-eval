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



from random import random
from collections import OrderedDict


THRESHOLD = 0.2



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

	with open("datasets/ml-latest-small/ratings_sample.csv", "w") as file:

		for userid in users_dict:

			rand = random()

			if rand <= threshold:
				count += 1
				for line in users_dict[userid]:
					file.write(line + "\n")



	print "Original # unique users:\t" +  str(len(users_dict))
	print "Sample   # unique users:\t" +  str(count)
	



###############################################
#  main
###############################################
if __name__ == '__main__':

    buildSample(THRESHOLD)

