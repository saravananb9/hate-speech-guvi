##### importing reguried packages
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
from gensim.models import Word2Vec
from gensim import corpora
from gensim.models import LdaModel
import torch
import os

# Load your dataset
metadata = pd.read_csv("E:\GUVI\Hate Speech Dataset\Annotations_Metadata.csv")

metadata['label_conversion'] = metadata['label'].apply(lambda x: 1 if x == 'hate' else 0)
# Step 2: Load Text Content
text_data = {}
text_files_dir = "E:/GUVI/Hate Speech Dataset/Text file"

for file_id in metadata["file_id"]:
    file_path = os.path.join(text_files_dir, f"{file_id}.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data[file_id] = file.read()

# Step 3: Data Integration
df = metadata.copy()
df["text"] = df["file_id"].map(text_data)

texts = df['text'].tolist()
labels = df['label'].tolist()

# Load the Danish hate speech detection model and tokenizer
model = BertForSequenceClassification.from_pretrained("alexandrainst/da-hatespeech-detection-base")
tokenizer = BertTokenizer.from_pretrained("alexandrainst/da-hatespeech-detection-base")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define a function to preprocess text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Removing stopwords
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    
    # Stemming and Lemmatization
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    
    return lemmatized_tokens

# Apply Word2Vec for word embeddings
tokenized_text = df['text'].apply(lambda x: preprocess_text(x))
word2vec_model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Apply topic modeling using LDA
dictionary = corpora.Dictionary(tokenized_text)
corpus = [dictionary.doc2bow(text) for text in tokenized_text]
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Define a function to classify text samples
def classify_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs.to(device)
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
    return predicted_label

# Apply classification to each row in the dataset and store results in a new column
df['hate_speech_label'] = df['text'].apply(classify_text)

# Add a new column indicating hate or not hate based on the 'hate_speech_label' column
df['hate_or_not_hate'] = df['hate_speech_label'].apply(lambda x: 'Hate' if x == 1 else 'Not Hate')

# Apply Word2Vec for word embeddings
tokenized_text = df['text'].apply(lambda x: preprocess_text(x))
word2vec_model = Word2Vec(tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Apply topic modeling using LDA
dictionary = corpora.Dictionary(tokenized_text)
corpus = [dictionary.doc2bow(text) for text in tokenized_text]
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Save the modified dataset to a new CSV file
df.to_csv("classify_dataset.csv", index=False)




################ model evaluation ####################
from sklearn.metrics import f1_score, precision_score, recall_score

# Assuming you have the ground truth labels in the 'actual_labels' column
actual_labels = df['label_conversion']

# Assuming you have the predicted labels from the hate speech detection model in the 'hate_speech_label' column
predicted_labels = df['hate_speech_label']

# Compute the F1 score, precision and recall scores
f1 = f1_score(actual_labels, predicted_labels, average='binary')
precision = precision_score(actual_labels, predicted_labels, average='binary')
recall = recall_score(actual_labels, predicted_labels, average='binary')

print("F1 Score:", f1)
print("Precision Score:", precision)
print("Recall Score:", recall)