#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           like_dislike_transform.py
#   author:         Ludovico Fabbri 1197400
#   description:    transform a rating dataset in like-dislike dataset 
#					default threshold = 3.5
#
#################################################################################



import argparse

THRESHOLD_DEFAULT = 3.0




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





def likeDislikeDatasetTransform(filepath, threshold):

	out_folder = parseOutputFolderPath(filepath)
	filename_no_ext = parseFileName(filepath)
	out_file = open(out_folder + filename_no_ext + "_binary.csv", "w")

	with open(filepath, "r") as file:

		for line in file:

			tokens = line.strip().split(",")
			score = float(tokens[2])

			# like
			if score >= threshold:
				out_file.write(tokens[0] + "\t" + tokens[1] + "\t" + "1" + "\t" + tokens[3] + "\n")

			# dislike
			else:
				out_file.write(tokens[0] + "\t" + tokens[1] + "\t" + "-1" + "\t" + tokens[3] + "\n")




	out_file.close()






















###############################################
#  main
###############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", type=str, help="file source path (string)")
    parser.add_argument("threshold", type=float, nargs="?", const=1, default=THRESHOLD_DEFAULT, choices=[2.0, 2.5, 3.0, 3.5, 4.0], help="binary threshold (float)")


    args = parser.parse_args()

    likeDislikeDatasetTransform(args.filepath, args.threshold)










