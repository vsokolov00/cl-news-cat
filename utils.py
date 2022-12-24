import numpy as np
import pandas as pd

df = pd.read_csv('rural_india_corpus.csv', sep=';')

def get_all_topics():
    all_topics = set('')
    index_by_topic = []
    df['topics'].fillna('', inplace=True)
    for topic in df['topics'].values:
        index_by_topic.append(np.array([topic.strip() for topic in topic.split(',') if topic != '']))
        all_topics.update(topic.split(','))

    all_topics = [topic.strip() for topic in all_topics if topic != '']

    return all_topics