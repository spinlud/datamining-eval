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
TRAIN_PREFIX = "_train_"
TEST_PREFIX = "_test_"
EXT = ".csv"
COSINE_SIMILARITY = "cos"
JACCARD_SIMILARITY = "jac"
PEARSON_CORR = "pears"
_P = 0.25





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






def collaborativeFilteringItemItem(filepath, similarity, hybrid=False):

    output_folder = parseOutputFolderPath(filepath)
    base_file_name = parseFileName(filepath)


    if hybrid:
    	out_file_base = "predictions_item_hybrid_" + similarity 
    else:
    	out_file_base = "predictions_item_" + similarity 

    out_file = open(output_folder + "output/" + out_file_base + EXT, "w")

    avg_rmse_dict = defaultdict(float)
    avg_mae_dict = defaultdict(float)


    # used only when hybrid = true ---------------------------------------------
    M_profiles = spio.loadmat(output_folder + "output/movie_profiles.mat")["M"]
    dots_profiles = M_profiles.dot(M_profiles.transpose())
    # --------------------------------------------------------------------------
    

    # for each fold
    for fold_index in xrange(1, NUM_FOLDS + 1):
    	
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
                        M_NORM[i, j] = M[i, j] - avg_movies[j]
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
        dots = M_NORM.transpose().dot(M_NORM)
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
                    top_k = topK_JaccardSimilarity(M_CSC, similarities, userid, movieid, hybrid, dots_profiles) 
                    
                else:
                    top_k = topK_CosineSimilarity(M_CSC, dots, similarities, userid, movieid, hybrid, dots_profiles)
                    


                r_xi = 0.0

                for k in _K:
                    num = 0.0
                    den = 0.0
                    b_xi = avg_users[userid] + avg_movies[movieid] - mu

                    for t in top_k[:k]:
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
    eval_out_path = output_folder + "output/" + out_file_base
    eval_out_path += "_hybrid_eval" if hybrid else "_eval"
    eval_out_path += EXT

    with open(eval_out_path, "w") as file:

        file.write("k" + "\t" + "RMSE" + "\t" + "MAE" + "\n")

        for k in _K:
            avg_rmse_dict[k] /= 5.0
            file.write(str(k) + "\t" + str(avg_rmse_dict[k]) + "\t" + str(avg_mae_dict[k]) + "\n")



        

    









def topK_CosineSimilarity(M_CSC, dots, similarities, userid, movieid, hybrid, dots_profiles):

    top_k = []
    item_sq_norm = dots[movieid, movieid]


    for j in M_CSC.getrow(userid).nonzero()[1]:

        if (j == movieid):
            continue

        if (movieid, j) in similarities:
            sim = similarities[ (movieid, j) ]
            top_k.append( (j, sim) )
            continue

        if (j, movieid) in similarities:
            sim = similarities[ (j, movieid) ]
            top_k.append( (j, sim) )
            continue

        if item_sq_norm == 0:
            similarities[ (movieid, j) ] = -1.0
            top_k.append( (j, -1.0) )
            continue

        j_sq_norm = dots[j, j]

        if j_sq_norm == 0:
            similarities[ (movieid, j) ] = -1.0
            top_k.append( (j, -1.0) )
            continue

        sim = dots[movieid, j] / math.sqrt(item_sq_norm * j_sq_norm)


        # if hybrid mode is selected, we compute linear combination between cf similarity and content-based similarity
        if hybrid:
        	cb_sim = 0.0
        	num = dots_profiles[movieid, j]
        	if num != 0:
        		den = math.sqrt(dots_profiles[movieid, movieid] * dots_profiles[j, j])
        		cb_sim = num / den

        	sim = _P * cb_sim + (1 - _P) * sim
        # --------------------------------------------------------------------

        similarities[ (movieid, j) ] = sim
        top_k.append( (j, sim) )


    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)

    return top_k







def topK_JaccardSimilarity(M_CSC, similarities, userid, movieid):
    
    top_k = []
    sim = 0

    # Jaccard similarity
    for j in M_CSC.getrow(userid).nonzero()[1]:

        if (j == movieid):
            continue

        if (movieid, j) in similarities:
            sim = similarities[ (movieid, j) ]

        elif (j, movieid) in similarities:
            sim = similarities[ (j, movieid) ]

        else:
            movieid_vector = set(M_CSC.getcol(movieid).nonzero()[0])
            j_vector = set(M_CSC.getcol(j).nonzero()[0])
            union = len(movieid_vector.union(j_vector))
            intersection = len(movieid_vector.intersection(j_vector))
            sim = intersection / float(union)
            similarities[ (j, movieid) ] = sim


        top_k.append((j, sim))

    # top 10 items similar to movieid rated by userid
    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)

    return top_k








###############################################
#  main
###############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", type=str, help="file source path (string)")
    parser.add_argument("--sim", type=str, nargs="?", const=1, default=PEARSON_CORR, choices=[COSINE_SIMILARITY, JACCARD_SIMILARITY, PEARSON_CORR], help="similarity measure (string)")
    parser.add_argument("--hybrid", type=bool, nargs="?", const=1, default=False, help="Mix collaborative-filtering and content-based similarities")
    parser.add_argument("--p", type=float, help="Linear combination factor")


    args = parser.parse_args()

    _P = args.p if args.p else _P


    collaborativeFilteringItemItem(args.filepath, args.sim, args.hybrid)










