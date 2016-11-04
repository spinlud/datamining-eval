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

import random
import numpy as np

import matplotlib.pyplot as plt
from numpy.random import normal

# import plotly.tools as tls
# import plotly.plotly as py  # tools to communicate with Plotly's server




AGE_CATEGORIES = [18, 24, 34, 44, 49, 55, 100]





def usersAVGS():

	users_avgs = defaultdict(float)
	users_counts = defaultdict(int)

	with open("u.data", "r") as file:
		for line in file:
			tokens = line.strip().split("\t")
			userid = int(tokens[0], 10)
			rating = float(tokens[2])

			users_avgs[userid] += rating
			users_counts[userid] += 1


	for userid in users_avgs:
		users_avgs[userid] /= float(users_counts[userid])


	return users_avgs	






def getMetadata():

	males = []
	females = []
	users_ages = defaultdict()
	users_occupations = defaultdict()


	# init dictionary for ages
	for cat in AGE_CATEGORIES:
		users_ages[cat] = []


	# init dictionary for occupations
	with open("u.occupation", "r") as file:
		for line in file:
			occ = line.strip()
			users_occupations[occ] = []

	# print len (users_occupations)


	with open("u.user", "r") as file:
		for line in file:
			tokens = line.strip().split("|")	

			userid = int(tokens[0], 10)
			age = int(tokens[1], 10)
			gender = tokens[2]
			occupation = tokens[3]
			zipcode = tokens[4]

			if gender == "M":
				males.append(userid)
			else:
				females.append(userid)


			users_occupations[occupation].append(userid)


			for cat in AGE_CATEGORIES:
				if age <= cat:
					users_ages[cat].append(userid)
					break


	return males, females, users_ages, users_occupations





	

	
	


def genderPlot(users_avgs, males, females, show=False):

	males_avg = 0.0
	females_avg = 0.0

	for userid in males:
		males_avg += users_avgs[userid]

	for userid in females:
		females_avg += users_avgs[userid]

	males_avg /= float(len(males))
	females_avg /= float(len(females))

	print males_avg, females_avg

	axes = plt.gca()
	axes.set_xlim([0,5])
	axes.set_ylim([1,5])
	axes.xaxis.grid(True) 	# Display x grid lines 
	axes.yaxis.grid(False) 	# Hide y grid lines 


	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off


	x1 = [1]
	y1 = [males_avg]

	x2 = [3]
	y2 = [males_avg]

	plt.bar(x1, y1, label="Male average ratings", color="Blue", width=1)
	plt.bar(x2, y2, label="Female average ratings", color="Purple", width=1)


	plt.title("Gender average movie ratings")
	plt.xlabel("Gender")
	plt.ylabel("Average movie ratings")

	plt.legend()	# show legend
	plt.grid()		# show grid

	plt.savefig("gender_avg_ratings.eps", format="eps", dpi=1000, bbox_inches='tight')

	if show:
		plt.show()





def agePlot(users_avgs, users_ages, show=False):

	avgs = defaultdict(float)
	counts = defaultdict(int)

	for cat in AGE_CATEGORIES:
		for userid in users_ages[cat]:
			avgs[cat] += users_avgs[userid]
			counts[cat] += 1


	for cat in avgs:
		avgs[cat] /= float(counts[cat])




	# plot 	


	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off

	
	i = 1
	for cat in AGE_CATEGORIES:
		label = str(cat)
		x = [i]
		y = [avgs[cat]]
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


	plt.title("Age average movie ratings")
	plt.xlabel("Age")
	plt.ylabel("Average movie ratings")

	plt.grid()		# show grid

	plt.savefig("age_avg_ratings.eps", format="eps", dpi=1000, bbox_inches='tight')

	if show:
		plt.show()






def occupationPlot(users_avgs, users_occupations, show=False):

	avgs = defaultdict(float)
	counts = defaultdict(int)

	for occ in users_occupations:
		for userid in users_occupations[occ]:
			avgs[occ] += users_avgs[userid]
			counts[occ] += 1


	for occ in avgs:
		avgs[occ] /= float(counts[occ])



	# plot 	


	plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off

	
	i = 1
	for occ in avgs:
		label = str(occ)
		x = [i]
		y = [avgs[occ]]
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



	plt.title("Occupation average movie ratings")
	plt.xlabel("Occupation")
	plt.ylabel("Average movie ratings")

	plt.grid()		# show grid

	plt.savefig('occupation_avg_ratings.eps', format='eps', dpi=1000, bbox_inches='tight')

	if show:
		plt.show()






	















###############################################
#  main
###############################################
if __name__ == '__main__':


	users_avgs = usersAVGS()

	males, females, users_ages, users_occupations = getMetadata()

	genderPlot(users_avgs, males, females, True)

	agePlot(users_avgs, users_ages, True)

	occupationPlot(users_avgs, users_occupations, True)






