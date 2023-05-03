import pandas as pd
import plotly.express as px
import numpy as np
import nltk
import plotly.graph_objs as go
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt')

input_file = 'filtered_resumes.csv'
df = pd.read_csv(input_file)

# Tokenize the resume text into sentences
df['Sentences'] = df['Resume_str'].apply(sent_tokenize)

# Concatenate all sentences across all resumes
all_sentences = df['Sentences'].explode().values.tolist()

# Build the term-sentence matrix using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
term_sentence_matrix = vectorizer.fit_transform(all_sentences)

# Apply SVD for rank-k approximation
k = 100
svd = TruncatedSVD(n_components=k)
rank_k_approximation = svd.fit_transform(term_sentence_matrix)

# Get the SVD matrices
U = svd.transform(term_sentence_matrix)  # U matrix
Sigma = np.diag(svd.singular_values_)    # ∑ matrix
VT = svd.components_                     # V^T matrix

# Rebuild the matrix A
A = U @ Sigma @ VT

# Compute term and sentence saliency scores (singular vectors)
term_saliency_scores = np.abs(svd.components_[0])  # Use absolute values of the first right singular vector
sentence_saliency_scores = rank_k_approximation.sum(axis=1)

# Function to extract top n keywords from a resume
def extract_top_keywords(resume, vectorizer, saliency_scores, n=10):
    tfidf_vector = vectorizer.transform([resume])
    tfidf_scores = tfidf_vector.toarray()[0]
    weighted_scores = tfidf_scores * saliency_scores
    top_n_indices = np.argsort(weighted_scores)[-n:]
    top_n_keywords = np.array(vectorizer.get_feature_names_out())[top_n_indices]
    return top_n_keywords[::-1]

# Function to extract top n sentences from a resume
def extract_top_sentences(sentences, vectorizer, svd_transformer, saliency_scores, n=10):
    sentence_vectors = vectorizer.transform(sentences)
    reduced_sentence_vectors = svd_transformer.transform(sentence_vectors)
    first_singular_value = saliency_scores[0]
    sentence_scores = reduced_sentence_vectors[:, 0] * first_singular_value
    top_n_indices = np.argsort(sentence_scores)[-n:]
    top_n_sentences = np.array(sentences)[top_n_indices]
    return top_n_sentences[::-1]


# Extract top keywords and sentences for each resume
top_n = 10
df['Top_Keywords'] = df['Resume_str'].apply(lambda x: extract_top_keywords(x, vectorizer, term_saliency_scores, top_n))
df['Top_Sentences'] = df['Sentences'].apply(lambda x: extract_top_sentences(x, vectorizer, svd, sentence_saliency_scores, top_n))

	
# Save the results to a CSV file
output_file = 'resumes_with_top_keywords_and_sentences.csv'
df[['ID', 'Top_Keywords', 'Top_Sentences']].to_csv(output_file, index=False)

# Print SVD matrices
print("U matrix (rank-k approximation):")
print(rank_k_approximation)
print("\n∑ matrix (singular values):")
print(svd.singular_values_)
print("\nV^T matrix:")
print(svd.components_)

print("\nOriginal Term-Sentence Matrix:")
print(term_sentence_matrix.toarray())
print("\nRank-k Approximated Term-Sentence Matrix:")
print(A)

print("Top Keywords and Sentences for each Resume:")
print(df[['ID', 'Top_Keywords', 'Top_Sentences']])

# Plot term saliency scores
term_fig = px.line(y=term_saliency_scores, title="Term Saliency Scores", labels=dict(index="Terms", value="Saliency Score"))
term_fig.show()

# Plot sentence saliency scores
sentence_fig = px.line(y=sentence_saliency_scores, title="Sentence Saliency Scores", labels=dict(index="Sentences", value="Saliency Score"))
sentence_fig.show()