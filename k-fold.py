#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           k-fold.py
#   author:         Ludovico Fabbri 1197400
#   description:    split input data for (stratified) k-fold cross validation
#					
#
#################################################################################


import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

# progress bar settings
widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]





def parseFileName(filepath):

	tokens = filepath.split("/")
	filename = tokens[len(tokens) - 1]
	tokens = filename.split(".")
	filename_no_ext = tokens[len(tokens) - 2]
	return filename_no_ext



def parseOutputFolderPath(filepath):

	output_path = ""
	tokens = filepath.split("/")

	for i in xrange(len(tokens) - 1):
		output_path += tokens[i] + "/"

	return output_path





def stratifiedCrossFold(filepath, num_folds, shuffle):


	file_data = []
	output_folder = parseOutputFolderPath(filepath)
	filename_no_ext = parseFileName(filepath)
	train_label = "_train_"
	test_label = "_test_"
	ext = ".csv"
	labels = np.arange(1, num_folds + 1)
	

	print "Loading input data.."
	with open(filepath, "r") as file:
		for line in file:
			file_data.append(line.strip())
	print "..done\n"



	n = len(file_data)


	X = np.arange(n)
	Y = np.repeat(0, n)

 	skf = StratifiedKFold(n_splits = num_folds, shuffle = shuffle)
 	skf.get_n_splits(X, Y)


 	split_result = skf.split(X, Y)

 	i = 0
 	print "Writing output files.."
 	for train_index, test_index in split_result:

 		train_path = output_folder + filename_no_ext + train_label + str(labels[i]) + ext
 		test_path = output_folder + filename_no_ext + test_label + str(labels[i]) + ext

 		with open(train_path, "w") as file:
 			for x in train_index:
 				file.write(file_data[x] + "\n")

 		with open(test_path, "w") as file:
 			for x in test_index:
 				file.write(file_data[x] + "\n")	

 		print str(len(train_index)), str(len(test_index))

 		i += 1		

	
	print "..done\n"














###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")
	parser.add_argument("num_folds", type=int, nargs="?", const=1, default=5, help="number of folds for k-fold cross-validation (integer)")
	parser.add_argument("shuffle", type=bool, nargs="?", const=1, default=True, help="randomize test sets indexes")

	args = parser.parse_args()

	stratifiedCrossFold(args.filepath, args.num_folds, args.shuffle)


	
	

