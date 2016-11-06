#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.sparse import lil_matrix
from collections import defaultdict
import math

import sys
import time

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer




def collaborativeFiltering():

    n = 943
    m = 1682
    S = '\t'
    PATH = 'datasets/ml-100k/u'
    TRN_EXT = '.base'
    TST_EXT = '.test'
    output_file = "hw3_out_small.txt"
    output_rmse_file = "hw3_rmse_small.txt"
    pbar_maxval = 20000

    # n = 71567
    # m = 65133
    # S = "::"
    # PATH = 'datasets/ml-10M100K/r'
    # TRN_EXT = '.train'
    # TST_EXT = '.test'
    # output_file = "hw3_out.txt"
    # output_rmse_file = "hw3_rmse.txt"
    # pbar_maxval = 2000015



    output = open(output_file, 'w')
    output_rmse = open(output_rmse_file, 'w')


    cross_val_rmse = defaultdict(float)

    # widgets = [Bar('>'), ' ', ETA(), ' ', ReverseBar('<')]
    widgets = ['Testing: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA(), ' ', FileTransferSpeed()]



    for num_test in xrange(1, 6):

        K = [2, 5, 10]
        mu = 0.0
        M = lil_matrix((n, m))
        avg_users = defaultdict(float)
        avg_items = defaultdict(float)
        rmse = defaultdict(float)
        path = PATH + str(num_test) + TRN_EXT
        jacc_sims = defaultdict()



        ''''''''''''''''''
        #   Training
        ''''''''''''''''''
        with open(path, 'r') as file:

            count = 0
            count_users = defaultdict(int)
            count_items = defaultdict(int)

            for line in file:
                tokens = line.split(S)
                if len(tokens) < 2:
                    continue
                user_id = int(tokens[0]) - 1
                item_id = int(tokens[1]) - 1
                rating = float(tokens[2])

                M[user_id, item_id] = rating
                avg_users[user_id] += rating
                avg_items[item_id] += rating
                mu += rating
                count_users[user_id] += 1
                count_items[item_id] += 1
                count += 1

            mu /= count

            for u in avg_users:
                avg_users[u] /= float(count_users[u])

            for i in avg_items:
                avg_items[i] /= float(count_items[i])


            print 'Training done.'



        ''''''''''''''''''
        #   Testing
        ''''''''''''''''''
        # init progress bar
        pbar = ProgressBar(widgets=widgets, maxval=pbar_maxval).start()

        path = PATH + str(num_test) + TST_EXT
        count = 0                               # number of ratings in the test set
        item_vectors = defaultdict()            # <item_id, set of indexes != 0>
        M_CSC = M.tocsc()



        for w in xrange(m):
            item_vectors[w] = set(M_CSC.getcol(w).nonzero()[0])

        with open(path, 'r') as file:
            for line in file:

                tokens = line.split(S)
                if len(tokens) < 2:
                    break
                user_id = int(tokens[0]) - 1
                item_id = int(tokens[1]) - 1
                rating = float(tokens[2])
                count += 1
                pbar.update(int(count))


                top_k = []
                sim = 0

                # Jaccard similarity
                for j in M.getrow(user_id).nonzero()[1]:
                    if (j == item_id):
                        continue

                    if (item_id < j):
                        if (item_id, j) in jacc_sims:
                            sim = jacc_sims[(item_id, j)]
                        else:
                            union = len(item_vectors[item_id].union(item_vectors[j]))
                            intersection = len(item_vectors[item_id].intersection(item_vectors[j]))
                            sim = intersection / float(union)
                            jacc_sims[(j, item_id)] = sim
                    else:
                        if (j, item_id) in jacc_sims:
                            sim = jacc_sims[(j, item_id)]
                        else:
                            union = len(item_vectors[item_id].union(item_vectors[j]))
                            intersection = len(item_vectors[item_id].intersection(item_vectors[j]))
                            sim = intersection / float(union)
                            jacc_sims[(j, item_id)] = sim

                    top_k.append((j, sim))

                # top 10 items similar to item_id rated by user_id
                top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:10]

                # prediction
                r_xi = 0.0

                for k in K:
                    num = 0.0
                    den = 0.0
                    b_xi = avg_users[user_id] + avg_items[item_id] - mu

                    for t in top_k[:k]:
                        curr_item_id = t[0]
                        sim = t[1]
                        b_xj = avg_users[user_id] + avg_items[curr_item_id] - mu

                        num += sim * (M[user_id, curr_item_id] - b_xj)
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

                    # update RMSE (1)
                    error = rating - r_xi
                    rmse[k] += error**2


                # if this is the first test, write result on file
                if num_test == 1:
                    output.write(tokens[0] + '\t' + tokens[1] + '\t' + str(r_xi) + '\n')


        # update RMSE and CROSS_VAL
        for k in K:
            rmse[k] = math.sqrt(rmse[k] / float(count))
            cross_val_rmse[k] += rmse[k]

        pbar.finish()


    # average on 5-fold
    for k in K:
        cross_val_rmse[k] /= 5.0
        output_rmse.write(str(k) + ': ' + str(cross_val_rmse[k]) + '\n')


    output.close()
    output_rmse.close()







''''''''''''''''''
#   main
''''''''''''''''''
if __name__ == '__main__':

    collaborativeFiltering()