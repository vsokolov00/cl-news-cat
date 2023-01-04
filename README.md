# Cross-lingual multi-label article categorization

## All statistics of the dataset is depicted in the `stats.ipynb` file.

## Step-by-step
1. Run the `scrape.py` script to update the `rural_india_corpus.csv` or use the pre-scraped version of the CSV file provided in this repo. The pre-scraped version may not contain all of the available articles.
2. Run the `prepare_dataset.py` to download the data to the directory specified by `--corpus` parameter and create the `data.csv` description file. Each folder in the corpus corresponds to an article, in this folder you can find translations of the article named according to the language codes (ISO 639). On the other hand, you can access the data on the BUT SGE server in `/mnt/scratch/tmp/xsokol15/corpus` and `/mnt/scratch/tmp/xsokol15/data.csv` (take into consideration that scratch disk is periodically formatted). The `data.csv` the following format:
`doc_id;path;lang;year;...categories...`

3. Before staring the training with the `train.py` script specify the path to the `data.csv` file in the configuration dictionary `config['data_desc_file']=PATH_TO_DATA.CSV`

4. `python train.py [-b]` can be run to train the model, if `-b` flag is set than the baseline model based on Logistic Regressions will be trained, otherwise MLP network with the specified configuration will be trained.
