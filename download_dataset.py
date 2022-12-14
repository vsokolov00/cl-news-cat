from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path

corpus_folder = Path("./corpus/")
corpus_desc_file = Path("./rural_india_corpus.csv")
df = pd.read_csv(corpus_desc_file, sep=';')


lang_prefix = {"English": "en", "Hindi": "hi", "Assamese": "as",
 "Chhattisgarhi": "hne", "Gujarati": "gu", "Kannada": "kn", "Punjabi": "pa",
 "Malayalam": "ml", "Marathi": "mr", "Tamil": "ta", "Urdu": "ur", "Telugu": "te",
 "Odia": "or", "Mizo": "lus", "Bengali": "bn"}

df.rename(columns=lang_prefix, inplace=True)

if not os.path.exists(corpus_folder):
    os.mkdir(corpus_folder)

for index, row in tqdm(df.iterrows(), total=len(df)):
    try:
        os.mkdir(corpus_folder / str(index))
    except FileExistsError:
        pass
    titles = []
    for lang in lang_prefix.values():
        if type(row[lang]) != float:
            url = row[lang]
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser')
            titles.append(lang + soup.find('span', id="article-gallery-title").text)
            text_paragraphs = soup.find_all('div', class_="paragraph-block")
            with open(corpus_folder / str(index) / f"{lang}.txt", "w") as f:
                for p in text_paragraphs:
                    text = p.find_all(text=True)
                    for p in text:
                        f.write(p.text)
            with open(corpus_folder / str(index) / f"labels.txt", "w") as f:
                f.write('\n'.join(row['topics'].split(',')))
            with open(corpus_folder / str(index) / f"title.txt", "w") as f:
                f.write(''.join(titles))