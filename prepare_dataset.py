from bs4 import BeautifulSoup
import requests
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from collections import Counter
import numpy as np
from utils import get_all_topics


pwd = Path(os.getcwd())
corpus_folder = Path("/mnt/scratch/tmp/xsokol15/corpus")
corpus_desc_file = Path(pwd / "rural_india_corpus.csv")

df = pd.read_csv(corpus_desc_file, sep=';')

lang_prefix = {"English": "en", "Hindi": "hi", "Assamese": "as",
 "Chhattisgarhi": "hne", "Gujarati": "gu", "Kannada": "kn", "Punjabi": "pa",
 "Malayalam": "ml", "Marathi": "mr", "Tamil": "ta", "Urdu": "ur", "Telugu": "te",
 "Odia": "or", "Mizo": "lus", "Bengali": "bn"}

all_topics = ['Little Takes', 'Adivasis', 'Invisible Women', 'Dalits', 'Visible Work', 'Small World', 'Things We Make',
    'Climate Change', 'Musafir', 'AudioZone', 'Getting There', 'The Rural in the Urban', 'Rural Sports', 'Environment',
    'Faces', 'PhotoZone', 'Things We Wear', 'Mosaic', 'VideoZone', 'PARI for Schools', 'We Are', 'PARI Anthologies',
    'Foot-Soldiers of Freedom', 'One-Offs', 'Women', 'Farming and its Crisis', 'Healthcare', 'Resource Conflicts',
    'Tongues', 'The Wild', 'Things We Do']

columns=['doc_id', 'path', 'lang', 'year'] + all_topics
out_df = pd.DataFrame(columns=columns)

df.rename(columns=lang_prefix, inplace=True)

if not os.path.exists(corpus_folder):
    os.mkdir(corpus_folder)

all_rows = []
try:
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            os.mkdir(corpus_folder / str(index))
        except FileExistsError:
            continue
        titles = []
        for lang in lang_prefix.values():
            if type(row[lang]) != float and type(row['topics']) != float:
                url = row[lang]
                r = requests.get(url)
                soup = BeautifulSoup(r.content, 'html.parser')
                try:
                    date = soup.find('div', class_='row-1').find('span', itemprop='datePublished').text
                except AttributeError:
                    continue
                titles.append(soup.find('span', id="article-gallery-title").text.strip())

                text_paragraphs = soup.find_all('div', class_="paragraph-block")
                with open(corpus_folder / str(index) / f"{lang}.txt", "w") as f:
                    f.write(soup.find('span', id="article-gallery-title").text) # title
                    for p in text_paragraphs:
                        text = p.find_all(text=True)
                        for p in text:
                            f.write(p.text)
                with open(corpus_folder / str(index) / f"labels.txt", "w") as f:
                    f.write('\n'.join(str(row['topics']).split(',')))

                labels = row['topics'].split(',')
                out_row = dict.fromkeys(get_all_topics(), 0)
                topics = {k: v+1 if k in labels else v for k, v in out_row.items()}
                out_row = {'doc_id': index, 'path': corpus_folder / str(index) / f"{lang}.txt", 'lang': lang, 'year': date.split(',')[1].strip(), **topics}
                all_rows.append(out_row)
except Exception:
    pass

out_df = pd.DataFrame(all_rows)

if os.path.exists(pwd / "data.csv"):
    df = pd.read_csv(pwd / "data.csv", sep=';')
    df_new = pd.concat([df, out_df], ignore_index=False)
    df_new.to_csv(pwd / "data.csv", sep=';', index=False)
else:
    out_df.to_csv(pwd / "data.csv", sep=';', index=False)
