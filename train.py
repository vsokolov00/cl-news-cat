import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from utils import get_all_topics, precompute_doc_embeddings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import wandb
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretrained_model = 'sentence-transformers/LaBSE'

if 'supergpu' in os.uname()[1]:
    pretrained_model = '/mnt/scratch/tmp/xsokol15/models/labse/'

# sentence_model = SentenceTransformer(pretrained_model, device=device)

topics_n = len(get_all_topics()) - 1
config = dict(
    hidden_layers=3,
    hidden_size=512,
    epochs=11,
    topics=topics_n,
    train_batch_size=128,
    test_batch_size=64,
    learning_rate=1e-2,
    weight_decay=1e-3,
    dropout=0.4,
    num_workers=0,
    wandb_run_desc="xlm",
    data_desc_file="data.csv")


from torch.nn.utils.rnn import pad_sequence #(1)
def custom_collate(data):
    inputs = [d[0] for d in data] #(3)
    inputs = pad_sequence(inputs, batch_first=True) #(4)
    labels = torch.tensor([d[1] for d in data], dtype=torch.float)

    return inputs, labels


class RuralIndiaDataset(Dataset):
    def __init__(self, data_desc_file, partition):
        super().__init__()
        tmp_df = pd.read_csv(precompute_doc_embeddings(), sep=';')
        tmp_df.drop(columns=['Invisible Women'], inplace=True)
        if partition == 'train':
            self.df = tmp_df[tmp_df['year'] > 2017]
        elif partition == 'test':
            self.df = tmp_df[tmp_df['year'] < 2018]
        #self.df = self.df.sample(frac=0.5, replace=False, random_state=123)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #text = self.df.iloc[idx].loc['text']
        test = self.df.iloc[idx].loc['embeddings'].replace('[', '').replace(']', '')
        embedding = torch.tensor(np.fromstring(test, dtype=float, sep=','), dtype=torch.float)
        topics = self.df.iloc[idx, 5:-2].tolist()

        #tokenized = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')["input_ids"].squeeze(0)
        #embedding = torch.tensor(sentence_model.encode([text])[0])
        return embedding, topics

    def get_pos_weight(self):
        tmp = torch.tensor(self.df.iloc[:, 5:-2].sum(axis=0).values, dtype=torch.float)
        tmp_col = self.df.columns[5:-2]

        for i in range(len(tmp)):
            print(f'Category: {tmp_col[i]} = {tmp[i]}')
        return torch.tensor((self.df.iloc[:, 5:-2].sum(axis=0).values / len(self.df))**(-1), dtype=torch.float)


class RuralIndiaModel(nn.Module):
    def __init__(self, input_size, n_topics, hidden_size, layers, dropout):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size, bias=True)
        self.do1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.fully_connected = nn.ModuleList()
        for _ in range(2):
            self.fully_connected.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.fully_connected.append(nn.Dropout(dropout))
            self.fully_connected.append(nn.ReLU())

        self.out = nn.Linear(hidden_size, n_topics, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #with torch.set_grad_enabled(False):
        #    x = self.embedding(x).pooler_output
        x = self.fc_in(x)
        x = self.relu(x)
        for f in self.fully_connected:
            x = f(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


def train(classifier, train_loader, validation_loader, criterion, optimizer, scheduler, epochs):
    wandb.watch(classifier, criterion, log="all", log_freq=10)
    classifier.to(device)

    vloss, prec, recall, f1 = validate(classifier, validation_loader, criterion=criterion)
    wandb.log({"Validation loss": vloss, "Precision": prec, "Recall": recall, "F1": f1})

    seen_ct = 0
    for epoch in range(1, epochs):
        avg_loss = 0
        # Training
        batch_ct = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for local_batch, local_labels in tepoch:
                batch_ct += 1
                tepoch.set_description(f"Epoch {epoch}")
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                seen_ct += len(local_batch)
                
                y_hat = classifier(local_batch)
                
                loss = criterion(y_hat, local_labels)
                avg_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=avg_loss / batch_ct)
                
                if ((batch_ct + 1) % 25) == 0:
                    wandb.log({"Training loss": avg_loss / batch_ct}, step=seen_ct)
            if epoch % 5 == 0:
                vloss, prec, recall, f1 = validate(classifier, validation_loader, criterion=criterion)
                wandb.log({"Validation loss": vloss, "Precision": prec, "Recall": recall, "F1": f1})
            scheduler.step(vloss)


def validate(model, loader, criterion):
    total_loss = 0
    total_correct = 0
    with torch.set_grad_enabled(False):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total = 0

        for local_batch, local_labels in loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            y_hat = model(local_batch)
                
            loss = criterion(y_hat, local_labels.float())
            total_loss += loss.item()

            predictions = (y_hat>0.5).float()

            total_tp += torch.sum(torch.logical_and(local_labels, predictions))
            fp_tmp = torch.sub(predictions, local_labels)
            fp_tmp[fp_tmp < 0] = 0
            total_fp += torch.sum(fp_tmp)
            total += torch.numel(local_labels)
            total_fn += torch.sum(torch.logical_and(torch.logical_xor(predictions, local_labels), local_labels))

            total_correct += torch.sum(torch.all(local_labels == predictions, dim=1))

        avg_vloss = total_loss / (len(loader))
        recall = total_tp.item() / (total_tp.item() + total_fn.item())
        precision = total_tp.item() / (total_tp.item() + total_fp.item())
        f1 = 2 * precision * recall / (precision + recall)
        wandb.log({"Test_Precision": precision, "Test_Recall": recall, "Test_F1": f1})
        print(f"Validation loss: {avg_vloss}, Accuracy (EMR): {total_correct / len(loader.dataset)}, Precision: {precision}, Recall: {recall}")
        return avg_vloss, precision, recall, f1


def test(model, loader):
    total_loss = 0
    total_correct = 0
    with torch.set_grad_enabled(False):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total = 0

        for local_batch, local_labels in loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            y_hat = model(local_batch)
                
            predictions = (y_hat>0.5).float()

            total_tp += torch.sum(torch.logical_and(local_labels, predictions))
            fp_tmp = torch.sub(predictions, local_labels)
            fp_tmp[fp_tmp < 0] = 0
            total_fp += torch.sum(fp_tmp)
            total += torch.numel(local_labels)
            total_fn += torch.sum(torch.logical_and(torch.logical_xor(predictions, local_labels), local_labels))

            total_correct += torch.sum(torch.all(local_labels == predictions, dim=1))

        recall = total_tp.item() / (total_tp.item() + total_fn.item())
        precision = total_tp.item() / (total_tp.item() + total_fp.item())
        f1 = 2 * precision * recall / (precision + recall)
        print(f"TEST:  Accuracy (EMR): {total_correct / len(loader.dataset)}, Precision: {precision}, Recall: {recall}")
        return precision, recall, f1

def train_logistic_reg(data_desc_file):
    train_dataset = RuralIndiaDataset(data_desc_file, partition='train')
    test_dataset = RuralIndiaDataset(data_desc_file, partition='test')

    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], num_workers=config['num_workers'], collate_fn=custom_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], num_workers=config['num_workers'], collate_fn=custom_collate, shuffle=True)

    inputs, classes = next(iter(train_loader))
    log_reg_models = []

    for i in range(classes.shape[1]):
        labels = classes[:, i].tolist()
        try:
            clf = LogisticRegressionCV(cv=5, random_state=123)
            clf.fit(inputs, labels)
            log_reg_models.append(clf)
        except ValueError:
            log_reg_models.append(None)
            continue
    print("Fitted")
    
    predictions = []
    total_score = 0
    inputs, classes = next(iter(test_loader))
    for i, model in enumerate(log_reg_models):
        if model is not None:
            prediction = model.predict(inputs)
            predictions.append(prediction)
            #total_score += model.score(inputs, classes[:, i].tolist())
        else:
            predictions.append(np.zeros(classes.shape[0]))
            continue
    predictions = torch.from_numpy(np.array(predictions).T)
    total_tp = torch.sum(torch.logical_and(classes, predictions))
    fp_tmp = torch.sub(predictions, classes)
    fp_tmp[fp_tmp < 0] = 0
    total_fp = torch.sum(fp_tmp)
    total_fn = torch.sum(torch.logical_and(torch.logical_xor(predictions, classes), classes))

    recall = total_tp.item() / (total_tp.item() + total_fn.item())
    precision = total_tp.item() / (total_tp.item() + total_fp.item())
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

def make(config):
    # Make the data
    dataset = RuralIndiaDataset(config.data_desc_file, partition='train')
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    test_dataset = RuralIndiaDataset(config.data_desc_file, partition='test')

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, num_workers=config.num_workers, collate_fn=custom_collate, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.test_batch_size, num_workers=config.num_workers, collate_fn=custom_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, num_workers=config.num_workers, collate_fn=custom_collate, shuffle=True)

    # Make the model
    classifier = RuralIndiaModel(input_size=768, n_topics=config.topics, layers=config.hidden_layers, hidden_size=config.hidden_size, dropout=config.dropout)

    # Make the loss and optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    pos_weight = dataset.get_pos_weight()
    pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, threshold=0.0001, threshold_mode='abs', verbose=True)

    return classifier, train_loader, validation_loader, test_loader, criterion, optimizer, scheduler

def model_pipeline(hyperparameters):

    # tell wandb to get started
    run_name = f"lr-{hyperparameters['learning_rate']}_dropout-{hyperparameters['dropout']}_layers-{hyperparameters['hidden_layers']}x{hyperparameters['hidden_size']}_run_{hyperparameters['wandb_run_desc']}"
    with wandb.init(project="zpja", config=hyperparameters, name=run_name):
      # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        classifier, train_loader, validation_loader, test_loader, criterion, optimizer, scheduler = make(config)
        train(classifier, train_loader, validation_loader, criterion, optimizer, scheduler, config.epochs)
        test(classifier, test_loader)


    return classifier


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('-b', '--baseline', action='store_true')
    args = argparse.parse_args()

    if args.baseline:
        train_logistic_reg(config['data_desc_file'])
    else:
        classifier = model_pipeline(config)
