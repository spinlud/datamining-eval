#!/usr/bin/python
# -*- coding: utf-8 -*-

#################################################################################
#
#   file:           cf_item_item.py
#   author:         Ludovico Fabbri 1197400
#   description:    item-item collaborative filtering
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


_N = 671 + 1
_M = 164979 + 1
NUM_FOLDS = 5
_K = [2, 5, 10, 20]
TOP_K = 10
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"
COSINE_SIMILARITY = "cos"
JACCARD_SIMILARITY = "jac"
PEARSON_CORR = "pears"





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






def collaborativeFilteringItemItem(filepath, similarity):

    output_folder = parseOutputFolderPath(filepath)
    base_file_name = parseFileName(filepath)

    out_file_base = "predictions_user_" + similarity 
    out_file = open(output_folder + "output/" + out_file_base + EXT, "w")

    avg_rmse_dict = defaultdict(float)
    avg_mae_dict = defaultdict(float)




    for fold_index in xrange(1, NUM_FOLDS + 1):
    # for fold_index in xrange(5, 6):    

        M = lil_matrix( (_N, _M) )
        M_NORM = lil_matrix( (_N, _M) )
        avg_users = defaultdict(float)
        avg_movies = defaultdict(float)

        rmse_dict = defaultdict(float)
        mae_dict = defaultdict(float)
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

                userid = int(tokens[0], 10)
                movieid = int(tokens[1], 10)
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


            pbar = ProgressBar(widgets=widgets, maxval=_N).start()


            if similarity == PEARSON_CORR:
                # normalize matrix
                for i in xrange(_N):
                    for j in M.getrow(i).nonzero()[1]:
                        M_NORM[i, j] = M[i, j] - avg_users[j]
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
        dots = M_NORM.dot(M_NORM.transpose())
        count = 0.0

        pbar = ProgressBar(widgets=widgets, maxval=21000).start()


        with open(test_path, "r") as file:

            for line in file:

                tokens = line.strip().split(",")
                

                userid = int(tokens[0], 10)
                movieid = int(tokens[1], 10)
                score = float(tokens[2])
                count += 1
                pbar.update(count)




                if similarity == JACCARD_SIMILARITY:
                    top_k = topK_JaccardSimilarity(M_CSC, similarities, userid, movieid) 
                    
                else:
                    top_k = topK_CosineSimilarity(M_CSC, dots, similarities, userid, movieid)
                    


                r_xi = 0.0



                for k in _K:
                    num = 0.0
                    den = 0.0
                    b_xi = avg_users[userid] + avg_movies[movieid] - mu
                    # b_xi = avg_users[userid]

                    for t in top_k[:k]:
                        y_id = t[0]
                        sim = t[1]
                        b_yi = avg_users[y_id] + avg_movies[movieid] - mu
                        # b_yi = avg_users[y_id]

                        num += sim * (M[y_id, movieid] - b_yi)
                        den += abs(sim)

                    if den != 0:
                        r_xi = b_xi + (num / float(den))
                    else:
                        r_xi = b_xi


                    # clamp prediction in [1,5]
                    if r_xi < 1:
                        r_xi = 1.0
                    if r_xi > 5:
                        r_xi = 5.0


                    # update rmse and mae
                    error = r_xi - score
                    rmse_dict[k] += error**2
                    mae_dict[k] = abs(error)

                # write predictions only for first test (fold)
                if (fold_index == 1):
                    out_file.write(tokens[0] + '\t' + tokens[1] + '\t' + str(r_xi) + '\n')




        # update rmse and mae
        for k in _K:
            rmse_dict[k] = math.sqrt(rmse_dict[k] / float(count))
            mae_dict[k] = math.sqrt(mae_dict[k] / float(count))
            avg_rmse_dict[k] += rmse_dict[k]
            avg_mae_dict[k] += mae_dict[k]

        out_file.close()


        pbar.finish()
        print "..done\n"
        print ""
        







    # average rmse on number of folds for each neighbour size and write to disk
    with open(output_folder + "output/" + out_file_base + "_eval" + EXT, "w") as file:

        file.write("k" + "\t" + "RMSE" + "\t" + "MAE" + "\n")

        for k in _K:
            avg_rmse_dict[k] /= 5.0
            file.write(str(k) + "\t" + str(avg_rmse_dict[k]) + "\t" + str(avg_mae_dict[k]) + "\n")



        

    









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
    sim = 0.0

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

    # top 10 users similar to userid that have rated movieid
    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:TOP_K]

    return top_k








###############################################
#  main
###############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", type=str, help="file source path (string)")
    parser.add_argument("--sim", type=str, nargs="?", const=1, default=PEARSON_CORR, choices=[COSINE_SIMILARITY, JACCARD_SIMILARITY, PEARSON_CORR], help="similarity measure (string)")


    args = parser.parse_args()


    collaborativeFilteringItemItem(args.filepath, args.sim)










