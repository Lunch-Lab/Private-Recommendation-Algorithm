import json
from gensim.models import Word2Vec
import re
import string

path = "C:/Users/yis82/OneDrive/Desktop/Lunch Lab/data/data"

# Load the preprocessed reviews
with open(path + "/processed_reviews.json", "r", encoding="utf-8") as f:
    processed_reviews = json.load(f)

cleaned_reviews = [
    [re.sub(r"[^가-힣A-Za-z0-9]", "", word) for word in review]
    for review in processed_reviews
]

# Train a Word2Vec model
model = Word2Vec(
    sentences=cleaned_reviews, vector_size=100, window=5, min_count=1, workers=4
)


def get_average_vector(review, model):
    """Generate the average word vector for a single review."""
    vectors = [model.wv[word] for word in review if word in model.wv]

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Assuming 'user_reviews' is a DataFrame with a 'user_id' and 'review' columns,
# where 'review' is a list of words for each review
user_vectors = {}

for user_id, reviews in user_reviews.groupby("user_id")["review"]:
    review_vectors = [get_average_vector(review, model) for review in reviews]
    user_vectors[user_id] = (
        np.mean(review_vectors, axis=0)
        if review_vectors
        else np.zeros(model.vector_size)
    )

# Extract word vectors from the model
word_vectors = model.wv.vectors

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Use t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=0)
vectors_2d = tsne.fit_transform(word_vectors)  # Apply t-SNE to the word vectors

# Plotting
plt.figure(figsize=(10, 10))
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

words = list(model.wv.index_to_key)
for i, word in enumerate(words[:100]):  # Limiting to first 100 words for clarity
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
    plt.text(vectors_2d[i, 0] + 0.03, vectors_2d[i, 1] + 0.03, word, fontsize=9)
plt.show()
