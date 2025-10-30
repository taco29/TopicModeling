import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)

def preprocess(text, lemmatizer, stop_words):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

if __name__ == "__main__":
    folder_path = r"C:\Users\Admin\Code\Social Listening\TopicModeling\data"
    docs = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    docs.append(text)
    print(f"Loaded {len(docs)} text files")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    texts = [preprocess(doc, lemmatizer, stop_words) for doc in docs]
    print(f"Cleaned {len(texts)} documents")

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in texts]

    number_topics = 4
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=number_topics,
        passes=15,
        iterations=50,
        alpha='auto',
        eta='auto',
        random_state=42
    )
    for i, topic in lda_model.show_topics(num_topics=number_topics, num_words=10, formatted=False):
        words = ", ".join([w for w, _ in topic])
        print(f"Topic {i + 1}: {words}")
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v'
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"\nCoherence Score (4 topics): {coherence_lda:.3f}")

    coherence_values = []
    topic_range = range(2, 21) 

    for k in topic_range:
        lda_k = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,       
            passes=15,
            iterations=50,
            alpha='auto',
            eta='auto',
            random_state=42
        )
        coh_model = CoherenceModel(
            model=lda_k,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coh_score = coh_model.get_coherence()
        coherence_values.append((k, coh_score))
        print(f"Computed coherence for k={k}: {coh_score:.3f}")

    x = [num for num, _ in coherence_values]
    y = [coh for _, coh in coherence_values]
    plt.plot(x, y, marker='o')
    plt.title("LDA Coherence Score by Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.show()