



from recsys.algorithm.factorize import SVD


# path = "datasets/ml-1m/ratings.dat"
path = "datasets/ml-latest-small/ratings_train_1.csv"

svd = SVD()
svd.load_data(filename=path,
            sep=',',
            format={'col':0, 'row':1, 'value':2, 'ids': float})

k = 30
svd.compute(k=k,
            min_values=10,
            pre_normalize=None,
            mean_center=True,
            post_normalize=True,
            savefile='/tmp/movielens')


# ITEMID1 = 1    # Toy Story (1995)
# ITEMID2 = 2355 # A bug's life (1998)

# print svd.similarity(ITEMID1, ITEMID2)




MIN_RATING = 1.0
MAX_RATING = 5.0

USERID = 1
ITEMID = 1129

print svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING)
print svd.predict(1953, 1, MIN_RATING, MAX_RATING)
# Predicted value 5.0

print svd.get_matrix().value(1953, 1)
# Real value 5.0