import os
from pdb import set_trace as st

from pyspark import SparkConf, SparkContext

def get_movies_csv(sc):
    """Create dependency graph to restructure the movies csv
    Also removes the header

    Args:
        sc (SparkContext): SparkContext instance used to read CSV

    Returns:
        PythonRDD: the movies_csv with each movie in the structure:
            (movieid, (name, set(genres))
    """
    mcsv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'datasets', 'movies.csv')
    movies_csv = sc.textFile(mcsv_path)
    movies_header = movies_csv.first()
    movies_csv = movies_csv.filter(lambda line: line != movies_header)
    return movies_csv.map(lambda line: line.split(',')).map(
        lambda x: (int(x[0]), (x[1], set(x[2].split('|')))))

def get_good_movies(sc, avg_rating):
    """Find all of the movies with an average rating above the avg_rating

    Args:
        sc (SparkContext): SparkContext instance used to read CSV
        avg_rating (float): the average rating of the user to determine
            which movies to recommend

    Returns:
        PythonRDD: the ratings_csv with each movie in the structure:
            (movie_id, avg_rating)
    """
    rcsv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'datasets', 'ratings.csv')
    ratings_csv = sc.textFile(rcsv_path)
    ratings_header = ratings_csv.first()
    ratings_csv = ratings_csv.filter(lambda line: line != ratings_header)
    restruct_ratings = ratings_csv.map(lambda line: line.split(',')).map(
        lambda x: (int(x[1]), (float(x[2]), 1)))
    red_ratings = restruct_ratings.reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1]))
    averaged_ratings = red_ratings.mapValues(lambda x: round(x[0]/x[1], 1))
    return averaged_ratings.filter(lambda rating: rating[1] >= avg_rating)

def grab_movie_tags(mid, mcsv):
    """Filter the given movie out with its tags

    Args:
        mid (String): movie_id to use as the source for recommendations
        mcsv (PythonRDD): movies_csv file to source genres for the given mid
    Return:
        movie: the actual movie tuple with its tags
    """
    return mcsv.filter(lambda movie: movie[0] == mid).first()

def rec_similar(mid, mcsv):
    """Recommend something similar to this movie using genres

    Args:
        mid (String): movie_id to use as the source for recommendations
        mcsv (PythonRDD): movies_csv file to source genres for the given mid
    Returns:
        PythonRDD: reference to dependency graph to filter RDD to just
            recommended movies
    """
    src_movie = grab_movie_tags(mid, mcsv)
    return mcsv.filter(
        lambda movie: movie[1][1].intersection(src_movie[1][1])).map(
            lambda x: (x[0], x[1][0]))

def recommender(user_id, movie_id, avg_rating):
    """Main function to figure out recommendations to give"""
    conf = SparkConf().setMaster('local[*]').setAppName('bds_project1')
    sc = SparkContext.getOrCreate(conf=conf)

    movies_csv = get_movies_csv(sc)
    above_avg = get_good_movies(sc, avg_rating)

    res = rec_similar(movie_id, movies_csv)

    recs = res.join(above_avg).takeSample(False, 5)
    return [movie[1] for movie in recs]


if __name__ == '__main__':
    givenMovieId = 31
    givenUserId = 1
    avgRating = 3
    print(recommender(givenUserId, givenMovieId, avgRating))
