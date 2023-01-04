import numpy as np
import pandas as pd
import os
import re
import textwrap
from tqdm import tqdm


from sentence_transformers import SentenceTransformer

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

def precompute_doc_embeddings(data_desc_fil='data.csv', out_df='ready_df.csv', sep=';', by_paragraph=True):
    """
    Precompute embeddings for all documents in the dataset.
    
    :param data_desc_fil: path to the file with the dataset description
    :param sep: separator in the dataset description file
    :return: embeddings for all documents in the dataset
    """
    if os.path.exists(out_df):
        return os.path.abspath(out_df)    


    df = pd.read_csv(data_desc_fil, sep=';')

    docs = []

    for _, row in df.iterrows():
        doc_file = open(row['path'], "r")
        text = doc_file.read().replace('\n', ' ')
        docs.append(text)
        doc_file.close()

    assert(len(docs) == len(df))
    df_out = df.copy()
    df_out['text'] = docs
    
    #model = SentenceTransformer('sentence-transformers/LaBSE')  # LaBSE for pre-embedding
    model = SentenceTransformer('/mnt/scratch/tmp/xsokol15/models/sentence-transformers_stsb-xlm-r-multilingual', device='cuda')
    model.max_seq_length = 512

    print("Encoding documents...")
    if not by_paragraph:
        embeddings = model.encode(df_out['text'].values)
        df_out['embeddings'] = embeddings.tolist()
    else:
        embeddings = []
        for doc in tqdm(df_out['text'].values, unit='docs'):
            doc = re.sub(r'\s+', ' ', doc)
            paragraphs = textwrap.wrap(doc, 512, break_long_words=False)
            
            embeddings_tmp = model.encode(paragraphs)
            embedding = np.mean(embeddings_tmp, axis=0)
            embeddings.append(embedding.tolist())
        df_out['embeddings'] = embeddings

    df_out.to_csv(out_df, sep=';')

    return os.path.abspath(out_df)
