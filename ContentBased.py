import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

data = pd.read_csv("movies.csv", encoding="latin-1", sep="\t", usecols=["title", "genres"])

data["genres"] = data["genres"].apply(lambda s: s.replace("|", " ").replace("-", "").lower())
data["title"] = data["title"].str.lower()

user_input = input("Enter a movie title you like: ").strip().lower()

# Handle incorrect or partial input
if user_input not in data["title"].values:
    matches = get_close_matches(user_input, data["title"], n=5, cutoff=0.5)
    if matches:
        # Display only the first closest match
        closest_match = matches[0]
        print(f"Did you mean: {closest_match}?")
        
        # Ask for confirmation
        user_confirmation = input(f"Please type 'yes' to confirm or 'no' to enter a new title: ").strip().lower()
        
        if user_confirmation == "yes":
            user_input = closest_match  # Auto-select the closest match
            print(f"Using movie title: {user_input}")
        elif user_confirmation == "no":
            user_input = input("Please enter a new movie title: ").strip().lower()
        else:
            print("Invalid input. Exiting...")
            exit()
    else:
        print("No close matches found.")
        exit()

vectorizer = TfidfVectorizer(ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform(data["genres"])
# print(vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
# print(tfidf_matrix.shape)

tfidf_matrix_dense = pd.DataFrame(tfidf_matrix.todense(), columns=vectorizer.get_feature_names_out(), index=data["title"])

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_dense = pd.DataFrame(cosine_sim, columns=data["title"], index=data["title"])

top_k = 30
relevant_data = cosine_sim_dense.loc[user_input].sort_values(ascending=False)[:top_k]

print("Recommended movies: ")
print(relevant_data)
