#############################################
# PROJECT: Hybrid Recommender System
# Data preparation
#############################################
import pandas as pd
pd.set_option("display.max_column", 5)
pd.set_option("display.width", 500)

movies = pd.read_csv("movie.csv")
ratings = pd.read_csv("Rating_Small.csv")
def dataframe_summary(df):
    summary = []
    cols = df.columns
    for col in cols:
        data_types = df[col].dtypes
        num_unique = df[col].nunique()
        sum_null = df[col].isnull().sum()
        summary.append([col, data_types, num_unique, sum_null])
    df_check = pd.DataFrame(summary)
    df_check.columns = ['columns', 'dtypes', 'nunique', 'sum_null']
    print(df_check)

dataframe_summary(movies)
dataframe_summary(ratings)
combined_df = pd.merge(ratings, movies, how="left", on="movieId")
#print(combined_df["title"].nunique())

# Removing rarely watched movies from dataframe
def supress_dataframe(df, variable_to_supress, supress_limits):
    for i in range(len(variable_to_supress)):
        supress_df = pd.DataFrame(df[variable_to_supress[i]].value_counts())
        supressed_list = supress_df.loc[supress_df[variable_to_supress[i]] > supress_limits[i]].index
        df = df.loc[df[variable_to_supress[i]].isin(supressed_list)]
    return df

# Supression limits are determined by looking at the mean and quantile data
print(combined_df["movieId"].value_counts().describe().T)
print(combined_df["userId"].value_counts().describe().T)
combined_df = supress_dataframe(combined_df, ["movieId", "userId"], [500, 40])
#print(combined_df.shape)
#print(user_movie_df.shape)
user_movie_df = combined_df.pivot_table(index="userId", columns="title", values="rating")


# Selecting a random user and finding the movies they watched
random_user = combined_df["userId"].sample(1).values[0]
random_user_df = user_movie_df.loc[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
print("They watched " + str(len(movies_watched)) + " movies")

# Finding other users that watched the same movies with random user and getting the index of them
movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
user_movie_count.columns = ["userId", "movie_count"]
# Finding users that watched as many as %60 of movies that random user wathced
users_same_movies = user_movie_count.loc[user_movie_count["movie_count"] > 0.6 * len(movies_watched), "userId"]


# Finding most similar users with the random users
final_df = pd.concat([movies_watched_df.loc[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])
#print(final_df[final_df.index == random_user])

corr_df = final_df.T.corr().unstack().sort_values(ascending=False).drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId1", "userId2"]
corr_df.reset_index(inplace=True)
top_users = corr_df.loc[(corr_df["userId1"] == random_user) & (corr_df["corr"] > 0.1), ["userId2", "corr"]].reset_index(drop=True)
top_users.columns = ["userId", "corr"]
top_users = top_users.loc[top_users["userId"] != random_user]
top_users_rating = top_users.merge(ratings, how="inner")


# Calculationg the weighted average score according to correlation with random user
top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]
recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()
top_recommendation_df = recommendation_df.loc[recommendation_df["weighted_rating"] > 1].sort_values("weighted_rating", ascending=False)

movies_to_be_recommend = top_recommendation_df["movieId"].head(5).tolist()
movie_names = movies.loc[movies["movieId"].isin(movies_to_be_recommend), "title"].tolist()
print(movie_names)
#############################################
# Item-Based Recommendation
#############################################
user = ratings["userId"].sample(1).values[0]
up_to_date_movie = combined_df.loc[combined_df["timestamp"] == combined_df.loc[combined_df["userId"] == user, "timestamp"].max(), "title"].values[0]
filtered_user_movie_df = user_movie_df[up_to_date_movie]
print("Recommended movies: " + str(user_movie_df.corrwith(filtered_user_movie_df).sort_values(ascending=False).drop(up_to_date_movie).head()))
