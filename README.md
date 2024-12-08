## Movie Recommendation System
This Python-based Movie Recommendation System uses movie genre similarity to suggest similar movies based on a user input. It leverages the TF-IDF Vectorizer from sklearn to calculate genre-based similarities and cosine similarity to rank the movies. If the user’s input is not an exact match, fuzzy matching with get_close_matches is used to suggest the closest titles.

# Features
Fuzzy Matching: If the movie title entered by the user doesn’t exactly match any movie in the dataset, it suggests the closest match.
Content-Based Recommendation: Recommends movies with similar genres to the one provided by the user.
User Interaction: Asks for user confirmation if the system finds multiple close matches.

# Requirements
To run this project, we need the following Python libraries:
pandas
scikit-learn
difflib

# Files
movies.csv: A dataset containing movie titles and their associated genres.
- title: The movie title.
- genres: A list of genres associated with the movie, separated by "|".
recommendation_system.py: The main Python script that runs the recommendation system.

# How It Works
- Load Data: The movie dataset (movies.csv) is loaded into a Pandas DataFrame, and genres are cleaned (replacing "|" with a space and removing hyphens).
- User Input: The program prompts the user to enter a movie title they like.
- Fuzzy Matching: If the entered title doesn't match any movie in the dataset, the system suggests similar movie titles using get_close_matches based on string similarity.
- TF-IDF Vectorization: The genres of all movies are transformed into numerical representations using the TF-IDF Vectorizer.
- Cosine Similarity: The cosine similarity between the genre vectors of the input movie and all other movies is calculated.
- Recommendation: The top 30 most similar movies are recommended to the user based on their genre similarity.

# Usage
- Place the movies.csv file in the same directory as the script.
- Run the Python script recommendation_system.py
- Enter a movie title when prompted.
- If there are close matches to your input, the system will ask for confirmation.
- After confirmation, it will display a list of recommended movies based on genre similarity.

# Modifications
- Threshold Adjustment: You can modify the cutoff parameter in get_close_matches to control how strict the fuzzy matching should be.
- Number of Recommendations: You can adjust the top_k value to recommend more or fewer movies.
