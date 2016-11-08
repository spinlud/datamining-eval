#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           hystograms.py
#   author:         Ludovico Fabbri 1197400
#   description:    
#
#
#################################################################################

from collections import defaultdict

import io
import random
import numpy as np

import matplotlib.pyplot as plt
from numpy.random import normal

# import plotly.tools as tls
# import plotly.plotly as py  # tools to communicate with Plotly's server




# def getGenres():
# 	unique_genres = []
# 	with open("datasets/ml-latest-small/unique_genres.csv", "r") as file:
# 		for line in file:
# 			unique_genres.add(line.strip())
# 	return unique_genres




# def avgsMovies():

# 	avgs = defaultdict(float)
# 	counts = defaultdict(int)

# 	with open("datasets/ml-latest-small/ratings.csv", "r") as file:
# 		for line in file:
# 			tokens = line.strip().split()
# 			movieid = int(tokens[1], 10)
# 			score = float(tokens[2])
# 			avgs[movieid] += score
# 			counts[movieid] += 1

# 	for movieid in avgs:
# 		avgs[movieid] /= float(counts[movieid])

# 	return avgs








# def getMetadata():

# 	avgs_movies = avgsMovies()

# 	genres = getGenres()

# 	genres_dict = defaultdict()
# 	years_dict = defaultdict()

# 	with open("datasets/ml-latest-small/")





	

	
	


# def genderPlot(users_avgs, males, females, show=False):

# 	males_avg = 0.0
# 	females_avg = 0.0

# 	for userid in males:
# 		males_avg += users_avgs[userid]

# 	for userid in females:
# 		females_avg += users_avgs[userid]

# 	males_avg /= float(len(males))
# 	females_avg /= float(len(females))

# 	print males_avg, females_avg

# 	axes = plt.gca()
# 	axes.set_xlim([0,5])
# 	axes.set_ylim([1,5])
# 	axes.xaxis.grid(True) 	# Display x grid lines 
# 	axes.yaxis.grid(False) 	# Hide y grid lines 


# 	plt.tick_params(
# 	    axis='x',          # changes apply to the x-axis
# 	    which='both',      # both major and minor ticks are affected
# 	    bottom='off',      # ticks along the bottom edge are off
# 	    top='off',         # ticks along the top edge are off
# 	    labelbottom='off') # labels along the bottom edge are off


# 	x1 = [1]
# 	y1 = [males_avg]

# 	x2 = [3]
# 	y2 = [males_avg]

# 	plt.bar(x1, y1, label="Male average ratings", color="Blue", width=1)
# 	plt.bar(x2, y2, label="Female average ratings", color="Purple", width=1)


# 	plt.title("Gender average movie ratings")
# 	plt.xlabel("Gender")
# 	plt.ylabel("Average movie ratings")

# 	plt.legend()	# show legend
# 	plt.grid()		# show grid

# 	plt.savefig("gender_avg_ratings.eps", format="eps", dpi=1000, bbox_inches='tight')

# 	if show:
# 		plt.show()


def yearsDict():
	years_dict = defaultdict()
	with io.open("datasets/ml-latest-small/movie_metadata.csv", "r", encoding="ISO-8859-1") as file:
		file.readline()
		for line in file:
			try:
				tokens = line.strip().split("\t")
				movieid = int(tokens[0], 10)
				year = tokens[3]
				years_dict[movieid] = year
			except:
				print line
	return years_dict




def genresDict ():
	genres_dict = defaultdict()
	with open("datasets/ml-latest-small/movies.csv", "r") as file:
		file.readline()
		for line in file:
			tokens = line.strip().split(",")
			movieid = int(tokens[0], 10)
			genres = [x.strip() for x in tokens[len(tokens) - 1].split("|")]
			genres_dict[movieid] = genres
	return genres_dict




def countryDict():
	country_dict = defaultdict()
	with io.open("datasets/ml-latest-small/movie_metadata.csv", "r", encoding="ISO-8859-1") as file:
		file.readline()
		for line in file:
			try:
				tokens = line.strip().split("\t")
				movieid = int(tokens[0], 10)
				country = tokens[8]
				country_dict[movieid] = country
			except:
				print line
	return country_dict





def genresPlot(show=False):

	avgs = defaultdict(float)
	counts = defaultdict(int)

	genres_dict = genresDict()


	with open("datasets/ml-latest-small/ratings.csv", "r") as file:
		# file.readline()
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[1], 10)
			score = float(tokens[2])

			for genre in genres_dict[movieid]:
				avgs[genre] += score
				counts[genre] += 1


	for genre in avgs:
		avgs[genre] /= float(counts[genre])



	# plot 	


	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off

	
	i = 1
	for genre in avgs:
		label = str(genre)
		x = [i]
		y = [avgs[genre]]
		plt.bar(x, y, label=label, color=np.random.rand(3,), width=1)

		i += 2


	axes = plt.gca()
	axes.xaxis.grid(True) 	# Display x grid lines 
	axes.yaxis.grid(False) 	# Hide y grid lines 	
	axes.set_ylim([1,5])
	axes.set_xlim([0,i])


	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=7, prop={'size':8})


	plt.title("Genre average movie ratings")
	plt.xlabel("Genre")
	plt.ylabel("Average movie ratings")

	plt.grid()		# show grid

	plt.savefig("genre_avg_ratings.eps", format="eps", dpi=1000, bbox_inches='tight')

	if show:
		plt.show()






def yearsPlot(show=False):

	years_dict = yearsDict()

	avgs = defaultdict(float)
	counts = defaultdict(int)


	with open("datasets/ml-latest-small/ratings.csv", "r") as file:
		# file.readline()
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[1], 10)
			score = float(tokens[2])

			if movieid in years_dict:
				year = years_dict[movieid]
				avgs[year] += score
				counts[year] += 1


	for year in avgs:
		avgs[year] /= float(counts[year])



	# plot 	

	# colors = []
	# interval = 1 / float(len(avgs))
	# print interval
	# r = 0.0
	# g = 0.0
	# b = 0.0
	# color = [r, g, b]
	# for i in xrange(len(avgs)):
	# 	color[0] += interval
	# 	color[1] += interval
	# 	color[2] += interval
	# 	colors.append(color)


	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off

	
	i = 1
	j = 0
	for year in sorted(avgs):
		label = str(year)
		x = [i]
		y = [avgs[year]]
		plt.bar(x, y, label=label, color=np.random.rand(3,), width=1)
		# plt.bar(x, y, label=label, color=colors[j], width=1)

		i += 2
		j += 1


	axes = plt.gca()
	axes.xaxis.grid(True) 	# Display x grid lines 
	axes.yaxis.grid(False) 	# Hide y grid lines 	
	axes.set_ylim([1,5])
	axes.set_xlim([0,i])


	ax = plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

 	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=7, prop={'size':8})



	plt.title("Year average movie ratings")
	plt.xlabel("Year")
	plt.ylabel("Average movie ratings")

	plt.grid()		# show grid

	plt.savefig('year_avg_ratings.eps', format='eps', dpi=1000, bbox_inches='tight')

	if show:
		plt.show()







def countryPlot(show=False):

	country_dict = countryDict()

	avgs = defaultdict(float)
	counts = defaultdict(int)

	with open("datasets/ml-latest-small/ratings.csv", "r") as file:
		# file.readline()
		for line in file:
			tokens = line.strip().split("\t")
			movieid = int(tokens[1], 10)
			score = float(tokens[2])

			if movieid in country_dict:
				country = country_dict[movieid]
				avgs[country] += score
				counts[country] += 1


	for country in avgs:
		avgs[country] /= float(counts[country])




	# plot

	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off

	
	i = 1

	color = np.random.rand(3,)
	
	for country in sorted(avgs):
		label = str(country)
		x = [i]
		y = [avgs[country]]
		plt.bar(x, y, label=label, color=color, width=1)

		i += 2
		


	axes = plt.gca()
	axes.xaxis.grid(True) 	# Display x grid lines 
	axes.yaxis.grid(False) 	# Hide y grid lines 	
	axes.set_ylim([0,6])
	axes.set_xlim([0,i])


	# ax = plt.subplot(111)
	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0 + box.height * 0.1,
 #                 box.width, box.height * 0.9])

 # 	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
 #          fancybox=True, shadow=True, ncol=7, prop={'size':8})



	plt.title("Country average movie ratings")
	plt.xlabel("Country")
	plt.ylabel("Average movie ratings")

	plt.grid()		# show grid

	plt.savefig('country_avg_ratings.eps', format='eps', dpi=1000, bbox_inches='tight')

	if show:
		plt.show()














	















###############################################
#  main
###############################################
if __name__ == '__main__':


	# genresPlot(True)

	# yearsPlot(True)

	countryPlot(True)




