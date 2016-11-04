#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           cf_user_user.py
#   author:         Ludovico Fabbri 1197400
#   description:    user-user collaborative filtering
#
#
#################################################################################


import io
import argparse
import math
import numpy as np
from scipy import io as spio
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from collections import defaultdict
from collections import OrderedDict

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

# progress bar settings
widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker()), ' ', ETA()]




TOP_K = 10
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"
COSINE_SIMILARITY = "cosine"
JACCARD_SIMILARITY = "jaccard"





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






def collaborativeFilteringItemItem(filepath, rows, cols, num_folds, similarity):

    output_folder = parseOutputFolderPath(filepath)
    base_file_name = parseFileName(filepath)
    out_path = output_folder + "output/predictions_user_user" + EXT
    out_file = open(out_path, "w")




    for fold_index in xrange(1, num_folds + 1):
    # for fold_index in xrange(5, 6):    

        M = lil_matrix( (rows, cols) )
        M_NORM = lil_matrix( (rows, cols) )
        avg_users = defaultdict(float)
        avg_movies = defaultdict(float)
        rmse_dict = defaultdict(float)
        mu = 0.0
        similarities = defaultdict(float)



        print "*** \t FOLD {0} \t ***".format(fold_index)

        #################################################################################
        #
        #	training phase
        #
        #################################################################################
        print "Training Phase.."
        train_path = output_folder + base_file_name + TRAIN_PREFIX + str(fold_index) + EXT


        with open(train_path, "r") as file:

            count = 0
            count_users = defaultdict(int)
            count_movies = defaultdict(int)

            for line in file:

                tokens = line.strip().split(",")

                userid = int(tokens[0], 10) - 1
                movieid = int(tokens[1], 10) - 1
                score = float(tokens[2])

                M[userid, movieid] = score
                avg_users[userid] += score
                avg_movies[movieid] += score
                count_users[userid] += 1
                count_movies[movieid] += 1
                mu += score
                count += 1




            mu /= count

            for i in avg_users:
                avg_users[i] /= float(count_users[i])


            for i in avg_movies:
                avg_movies[i] /= float(count_movies[i])


            pbar = ProgressBar(widgets=widgets, maxval=rows).start()

            # normalize matrix by user mean
            for i in xrange(rows):
                for j in M.getrow(i).nonzero()[1]:
                    M_NORM[i, j] = M[i, j] - avg_users[i]
                pbar.update(i)


            pbar.finish()
            print "..done"





        #################################################################################
        #
        #	test phase
        #
        #################################################################################
        
        print "Test Phase.."
        test_path = output_folder + base_file_name + TEST_PREFIX + str(fold_index) + EXT
        

        print test_path

        M_CSC = M.tocsc()
        # M_NORM_TRANSP = M_NORM.transpose()
        dots = M_NORM.dot(M_NORM.transpose())
        count = 0.0

        pbar = ProgressBar(widgets=widgets, maxval=21000).start()


        with open(test_path, "r") as file:

            for line in file:

                tokens = line.strip().split(",")
                

                userid = int(tokens[0], 10) - 1
                movieid = int(tokens[1], 10) - 1
                score = float(tokens[2])
                count += 1
                pbar.update(count)




                if similarity == COSINE_SIMILARITY:
                    top_k = topK_CosineSimilarity(M_CSC, dots, similarities, userid, movieid)

                else:
                    top_k = topK_JaccardSimilarity(M_CSC, similarities, userid, movieid) 
                    # print str(len(similarities))                   


                r_xi = 0.0


                num = 0.0
                den = 0.0
                b_xi = avg_users[userid] + avg_movies[movieid] - mu

                for t in top_k:
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


                # update rmse
                error = score - r_xi
                rmse_dict[fold_index] += error**2

                # write predictions only for first test (fold)
                if (fold_index == 1):
                    out_file.write(tokens[0] + '\t' + tokens[1] + '\t' + str(r_xi) + '\n')


            # update rmse
            rmse_dict[fold_index] = math.sqrt(rmse_dict[fold_index] / float(count))

            out_file.close()


        pbar.finish()
        print "..done\n"
        print ""
        







    # average rmse on number of folds
    avg_rmse = 0.0
    for key in rmse_dict:
        avg_rmse += rmse_dict[key]
    avg_rmse /= float(num_folds)


    # write rmse
    with open(output_folder + "output/rmse_user_user" + EXT, "w") as file:
        file.write(str(TOP_K) + "\t" + str(avg_rmse))

    









def topK_CosineSimilarity(M_CSC, dots, similarities, userid, movieid):

    top_k = []
    user_sq_norm = dots[userid, userid]

    # find users most similar to userid that have rated movieid
    for i in M_CSC.getcol(movieid).nonzero()[0]:

        if (i == userid):
            continue

        if (userid, i) in similarities:
            sim = similarities[ (userid, i) ]
            top_k.append( (i, sim) )
            continue

        if (i, userid) in similarities:
            sim = similarities[ (i, userid) ]
            top_k.append( (i, sim) )
            continue

        if user_sq_norm == 0:
            similarities[ (userid, i) ] = -1.0
            top_k.append( (i, -1.0) )
            continue

        i_sq_norm = dots[i, i]

        if i_sq_norm == 0:
            similarities[ (userid, i) ] = -1.0
            top_k.append( (i, -1.0) )
            continue

        sim = dots[userid, i] / math.sqrt(user_sq_norm * i_sq_norm)
        similarities[ (userid, i) ] = sim
        top_k.append( (i, sim) )


    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:TOP_K]

    return top_k







def topK_JaccardSimilarity(M_CSC, similarities, userid, movieid):
    
    top_k = []
    sim = 0

    # Jaccard similarity
    for i in M_CSC.getcol(movieid).nonzero()[0]:

        if (i == userid):
            continue

        if (userid, i) in similarities:
            sim = similarities[ (userid, i) ]

        elif (i, userid) in similarities:
            sim = similarities[ (i, userid) ]

        else:
            userid_vector = set(M_CSC.getrow(userid).nonzero()[1])
            i_vector = set(M_CSC.getrow(i).nonzero()[1])
            union = len(userid_vector.union(i_vector))
            intersection = len(userid_vector.intersection(i_vector))
            sim = intersection / float(union)
            similarities[ (i, userid) ] = sim


        top_k.append((j, sim))

    # top 10 items similar to movieid rated by userid
    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:TOP_K]

    return top_k








###############################################
#  main
###############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", type=str, help="file source path (string)")
    parser.add_argument("utility_rows", type=int, help="utility matrix # rows (int)")
    parser.add_argument("utility_columns", type=int, help="utility matrix # columns (int)")
    parser.add_argument("num_folds", type=int, help="cross-validation # folds (int)")
    parser.add_argument("similarity", type=str, nargs="?", const=1, default=COSINE_SIMILARITY, help="similarity measure (string)")


    args = parser.parse_args()

    collaborativeFilteringItemItem(args.filepath, args.utility_rows, args.utility_columns, args.num_folds, args.similarity)










