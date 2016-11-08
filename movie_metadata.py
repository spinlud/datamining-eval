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
import csv
import requests
import argparse
import pprint
import json
from collections import defaultdict
from collections import OrderedDict


BASE_URL = "http://www.omdbapi.com/?"
ID_PREFIX = "tt"
SEP = ","
FIELDS = ["Title", "Year", "Director", "Actors", "Genre", "Language", "Country", "Awards", "Plot"]

PP = pprint.PrettyPrinter(indent=4)


# http://www.omdbapi.com/?t=Star+Wars&y=1977&plot=full&r=json



def parseOutputFolderPath(filepath):

	output_path = ""
	tokens = filepath.split("/")

	for i in xrange(len(tokens) - 1):
		output_path += tokens[i] + "/"

	return output_path








def getMovieMetadata():

	# tagged_movies = getTaggedMovies()

	with open("datasets/ml-20m/links.csv", "r") as infile:
		with io.open("datasets/ml-20m/movie_metadata.csv", "w", encoding="ISO-8859-1") as outfile:

			header = u"movieid" + "\t" + "imdbId" + "\t"
			for field in FIELDS:
				header += field + "\t"
			header = header.strip() + "\n"
			outfile.write(header)
			
			infile.readline()		# skip header

			i = 1
			errors = 0

			for line in infile:
				tokens = line.strip().split(",")
				movieid = int(tokens[0], 10)
				omdb_id = tokens[1]


				if len(omdb_id) == 0:
					continue


				try:
					url = BASE_URL + "i={0}{1}&type=movie&plot=full&r=json".format(ID_PREFIX, omdb_id)
					response = requests.get(url)

					json_obj = json.loads(response.text)

					s = u""
					s += str(movieid) + "\t" + omdb_id + "\t"
					
					for field in FIELDS:
						s += json_obj[field] + "\t"

					s = s.strip() + "\n"
					outfile.write(s)

					


				except Exception as e:
					print e
					s = u""
					s += str(movieid) + "\t" + omdb_id + "\n" 
					outfile.write(s)
					errors += 1

			

				print "write " + str(i)
				i += 1


			print "Errors: " + errors








# def test(filepath):

# 	src_folder = parseOutputFolderPath(filepath)
# 	out_file = io.open(src_folder + "output/metadata_full.csv", "w", encoding="ISO-8859-1")


# 	old_dict = OrderedDict()
# 	new_dict = OrderedDict()

# 	with io.open(filepath, "r", encoding="ISO-8859-1") as file:

# 		for line in file:
# 			tokens = line.strip().split("|")
# 			movieid = int(tokens[0], 10)
# 			title = tokens[1]
# 			if title[-1:] == ")" and title[-2:-1].isdigit():
# 				title = title[:-7]
# 			title = title.replace(", The", "")

# 			old_dict[title] = int(line[0], 10)


# 	print str(len(old_dict.keys()))




# 	with io.open("datasets/ml-latest-small/latest_small_metadata_full.csv", "r", encoding="ISO-8859-1") as file:
# 		file.readline()		# skip header
# 		for line in file:
# 			tokens = line.strip().split("\t")
# 			try:
# 				movieid = int(tokens[0], 10)
# 				title = tokens[3]
# 				new_dict[title] = movieid
# 			except:
# 				print line


# 	total = 0
# 	count = 0

# 	for key in old_dict:
# 		total += 1
# 		if key in new_dict:
# 			count += 1

# 	print "{0}/{1}".format(count, total)







# def getMetadata(filepath):

# 	out_folder = parseOutputFolderPath(filepath)
# 	out_file = io.open(out_folder + "latest_small_metadata_full.csv", "w", encoding="ISO-8859-1")

# 	with open(filepath, "r") as file:

# 		line = file.readline()	# skip header
# 		headers = line.strip().split(SEP)
# 		out_header = u""
# 		for header in headers:
# 			out_header += header + "\t"
# 		for field in FIELDS:
# 			out_header += field + "\t"
# 		out_header = out_header.strip() + "\n"
# 		out_file.write(out_header)

		
# 		i = 1

# 		for line in file:

# 			tokens = line.strip().split(SEP)
			

# 			try:
# 				url = BASE_URL + "i={0}{1}&type=movie&plot=full&r=json".format(ID_PREFIX, tokens[1])
# 				response = requests.get(url)

# 				json_obj = json.loads(response.text)

# 				s = u""
# 				s += tokens[0] + "\t" + tokens[1] + "\t" + tokens[2] + "\t"
				
# 				for field in FIELDS:
# 					s += json_obj[field] + "\t"

# 				s = s.strip() + "\n"
# 				out_file.write(s)

				


# 			except Exception as e:
# 				print e
# 				s = u""
# 				s += tokens[0] + "\t" + tokens[1] + "\t" + tokens[2]
# 				out_file.write(s)

		

# 			print "write " + str(i)
# 			i += 1
		
		

		


		






















###############################################
#  main
###############################################
if __name__ == '__main__':

	# parser = argparse.ArgumentParser()

	# parser.add_argument("filepath", type=str, help="file source path (string)")
	

	# args = parser.parse_args()

	# getMetadata(args.filepath)

	# test(args.filepath)



	getMovieMetadata()





