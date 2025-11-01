import os
import re
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text, stop_words):
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

if __name__ == "__main__":

    docs = []
    folder_path = r"C:\Users\Admin\Code\Social Listening\TopicModeling\data"

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read().strip().lower()
                text = preprocess(text, stop_words)
                if text:
                    docs.append(text)
    print(f"Loaded {len(docs)} text files")

    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        verbose=True,
        language="english",
        min_topic_size=3,
        #nr_topics="auto",
    )

    topics, _ = topic_model.fit_transform(docs)

    info = topic_model.get_document_info(docs)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(info)

    for topic_id in info["Topic"].unique():
        if topic_id != -1:
            print(f"\n Topic {topic_id}:")
            words = [word for word, _ in topic_model.get_topic(topic_id)]
            print(", ".join(words))
    
    #coherence score
    tokenized_docs = [doc.split() for doc in docs]

    topics_words = [
        [word for word, _ in topic_model.get_topic(topic_id)]
        for topic_id in topic_model.get_topic_freq().Topic if topic_id != -1
    ]

    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    coherence_model = CoherenceModel(
        topics=topics_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    print(f"Topic Coherence Score (c_v): {coherence_score:.4f}")
