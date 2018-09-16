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
    movies_csv = sc.textFile('./movies.csv')
    movies_header = movies_csv.first()
    movies_csv = movies_csv.filter(lambda line: line != movies_header)
    return movies_csv.map(lambda line: line.split(',')).map(
        lambda x: (int(x[0]), (x[1], set(x[2].split('|')))))

def get_average_ratings(sc):
    """Create dependency graph to calculate all of the movies' average ratings

    Args:
        sc (SparkContext): SparkContext instance used to read CSV

    Returns:
        PythonRDD: the ratings_csv with each movie in the structure:
            (movieid, (rating, num_ratings_added))
    """
    ratings_csv = sc.textFile('./ratings.csv')
    ratings_header = ratings_csv.first()
    ratings_csv = ratings_csv.filter(lambda line: line != ratings_header)
    restruct_ratings = ratings_csv.map(lambda line: line.split(',')).map(
        lambda x: (int(x[1]), (float(x[2]), 1)))
    red_ratings = restruct_ratings.reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1]))
    return red_ratings.mapValues(lambda x: round(x[0]/x[1], 1))

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
    #st() # separate thread creating name errors?
    src_movie = grab_movie_tags(mid, mcsv)
    return mcsv.filter(
        lambda movie: movie[1][1].intersection(src_movie[1][1])).map(
            lambda x: (x[0], x[1][0]))

def rec_different(mid, mcsv):
    """Recommend something different to this movie using genres

    Args:
        mid (String): movie_id to use as the source for recommendations
        mcsv (PythonRDD): movies_csv file to source genres for the given mid
    Returns:
        PythonRDD: reference to dependency graph to filter list to recommended
            movies
    """
    src_movie = grab_movie_tags(mid, mcsv)
    return mcsv.filter(
        lambda movie: not movie[1][1].intersection(src_movie[1][1])).map(
            lambda x: (x[0], x[1][0]))

def rec_something(mid, rating, mcsv, averages):
    """Decide whether to recommend different or similar movies based on the
    user's rating compared to the average rating

    Args:
        mid (String): movie_id to use as the source for recommendations
        rating (String): rating the user gave for the mid
        mcsv (PythonRDD): movies_csv file to source genres for the given mid
        averages (PythonRDD): processed csv file for all of the movies' average
            ratings
    Returns:
        PythonRDD: reference to dependency graph to perform action
    """
    avg_rating = averages.filter(lambda x: x[0] == mid).first()[1]
    return rec_different(mid, mcsv) if rating < avg_rating else rec_similar(mid, mcsv)

def recommender(user_id, movie_id, avg_rating):
    """Main function to figure out recommendations to give"""
    conf = SparkConf().setMaster('local[*]').setAppName('bds_project1')
    sc = SparkContext.getOrCreate(conf=conf)

    movies_csv = get_movies_csv(sc)
    average_ratings = get_average_ratings(sc)

    if avg_rating >= 4:
        res = rec_similar(movie_id, movies_csv)
    elif avg_rating < 2:
        res = rec_different(movie_id, movies_csv)
    else:
        res = rec_something(movie_id, avg_rating, movies_csv, average_ratings)

    # print(res.takeSample(False, 5))
    print(res.leftOuterJoin(average_ratings).take(10))
    return res.first()


if __name__ == '__main__':
    givenMovieId = 31
    givenUserId = 1
    avgRating = 4
    print(recommender(givenUserId, givenMovieId, avgRating))
