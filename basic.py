import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

# Download the NLTK stopwords and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Read the input CSV file
input_file = 'filtered_resumes.csv'
df = pd.read_csv(input_file)

# Function to preprocess and tokenize a resume
def preprocess_and_tokenize(resume):
    # Convert to lowercase and remove special characters
    resume = resume.lower()
    resume = nltk.re.sub('[^a-z]+', ' ', resume)

    # Tokenize the resume
    tokens = word_tokenize(resume)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

# Preprocess and tokenize the resumes in the DataFrame
df['TokenizedResume'] = df['Resume_str'].apply(preprocess_and_tokenize)

# Example: Calculate word frequencies for the first resume
freq_dist = FreqDist(df['TokenizedResume'][0])

# Display the 10 most common words in the first resume
print(freq_dist.most_common(10))
