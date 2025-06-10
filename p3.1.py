import gensim
import nltk
import string
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

nltk.download('punkt')
nltk.download('stopwords')

medical_corpus = [
    "Diabetes is a chronic disease that affects the way the body processes blood sugar.",
    "Hypertension, or high blood pressure, can lead to heart disease and stroke.",
    "The patient was diagnosed with pneumonia and prescribed antibiotics.",
    "Insulin therapy is commonly used for type 1 diabetic patients.",
    "Cardiovascular diseases are the leading cause of death worldwide."
]

stop_words = set(stopwords.words('english'))
processed_corpus = [
    [word for word in word_tokenize(sentence.lower()) if word not in stop_words and word not in string.punctuation]
    for sentence in medical_corpus
]

model = Word2Vec(sentences=processed_corpus, vector_size=300, window=5, min_count=1, workers=4)

print("Vocabulary:", model.wv.index_to_key)

print("Similar words to 'diabetes':", model.wv.most_similar("diabetes"))

res = model.wv.most_similar(positive=["hypertension", "heart"], negative=["stroke"])
print("Result of (hypertension + heart - stroke):", res)

words = model.wv.index_to_key
word_vecs = [model.wv[word] for word in words]

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vecs)

plt.figure(figsize=(10, 6))
for i, word in enumerate(words):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
    plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
plt.title("Word Embeddings Visualization (PCA)")
plt.show()