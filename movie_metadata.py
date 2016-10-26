#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           movie_metadata.py
#   author:         Ludovico Fabbri 1197400
#   description:    get movies metadata from https://www.omdbapi.com
#					
#
#################################################################################


import io
import requests
import argparse
import pprint
import json


BASE_URL = "http://www.omdbapi.com/?"
ID_PREFIX = "tt"
SEP = ","
FIELDS = ["Title", "Year", "Director", "Actors", "Genre", "Language", "Country", "Plot"]

PP = pprint.PrettyPrinter(indent=4)






def parseOutputFolderPath(filepath):

	output_path = ""
	tokens = filepath.split("/")

	for i in xrange(len(tokens) - 1):
		output_path += tokens[i] + "/"

	return output_path




def getMetadata(filepath):

	out_folder = parseOutputFolderPath(filepath)
	out_file = io.open(out_folder + "metadata.csv", "w", encoding="ISO-8859-1")

	with open(filepath, "r") as file:

		line = file.readline()	# skip header
		headers = line.strip().split(SEP)
		out_header = u""
		for header in headers:
			out_header += header + "\t"
		for field in FIELDS:
			out_header += field + "\t"
		out_header = out_header.strip() + "\n"
		out_file.write(out_header)

		
		i = 1

		for line in file:

			tokens = line.strip().split(SEP)
			

			try:
				url = BASE_URL + "i={0}{1}&type=movie&plot=short&r=json".format(ID_PREFIX, tokens[1])
				response = requests.get(url)

				json_obj = json.loads(response.text)

				s = u""
				s += tokens[0] + "\t" + tokens[1] + "\t" + tokens[2] + "\t"
				
				for field in FIELDS:
					s += json_obj[field] + "\t"

				s = s.strip() + "\n"
				out_file.write(s)

				


			except Exception as e:
				print e
				s = u""
				s += tokens[0] + "\t" + tokens[1] + "\t" + tokens[2]
				out_file.write(s)

		

			print "write " + str(i)
			i += 1
		
		

		


		






















###############################################
#  main
###############################################
if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("filepath", type=str, help="file source path (string)")
	

	args = parser.parse_args()

	getMetadata(args.filepath)







