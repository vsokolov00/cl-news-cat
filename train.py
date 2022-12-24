import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import get_all_topics
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

wandb.login()

df = pd.read_csv('data.csv', sep=';')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LaBSE = SentenceTransformer('LaBSE', device=device)

config = dict(
    hidden_size=512,
    epochs=10,
    topics=19,
    train_batch_size=4,
    test_batch_size=2,
    learning_rate=0.001,
    data_desc_file="data.csv")


class RuralIndiaDataset(Dataset):
    def __init__(self, data_desc_file, partition):
        super().__init__()
        tmp_df = pd.read_csv(data_desc_file, sep=';')
        if partition == 'train':
            self.df = tmp_df[tmp_df['year'] > 2017]
        elif partition == 'test':
            self.df = tmp_df[tmp_df['year'] < 2018]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        doc_file = open(row['path'], "r")
        data = LaBSE.encode([doc_file.read()])[0]
        doc_file.close() 
        topics = row.iloc[4:].to_numpy(dtype=float)

        return data, topics


class RuralIndiaModel(nn.Module):
    def __init__(self, input_size, n_topics, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, n_topics, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x


def train(classifier, train_loader, validation_loader, criterion, optimizer, epochs):
    wandb.watch(classifier, criterion, log="all", log_freq=10)
    classifier.to(device)

    seen_ct = 0
    for epoch in range(1, epochs):
        avg_loss = 0
        # Training
        batch_ct = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            batch_ct += 1
            for local_batch, local_labels in tepoch:
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

                # Report metrics every 25th batch
                if ((batch_ct + 1) % 3) == 0:
                    print("Wandb logging")
                    wandb.log({"epoch": epoch, "loss": loss}, step=seen_ct)
            #train_losses.append(avg_loss / len(train_loader))

            if epoch % 1 == 0:
                vloss = validate(classifier, validation_loader)
                #valid_losses.append(vloss)

def validate(model, loader, criterion=nn.BCELoss()):
    total_loss = 0
    total_correct = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            y_hat = model(local_batch)

            loss = criterion(y_hat, local_labels.float())
            total_loss += loss.item()

            predictions = (y_hat>0.5).float()
            
            total_correct += torch.sum(torch.all(local_labels == predictions, dim=1))

        avg_vloss = total_loss / (len(loader))
        print(f"Validation loss: {avg_vloss}, Accuracy (EMR): {total_correct / len(loader.dataset)}")
        return avg_vloss

def make(config):
    # Make the data
    dataset = RuralIndiaDataset(config.data_desc_file, partition='train')
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    test_dataset = RuralIndiaDataset('data.csv', partition='test')

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.test_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=True)

    # Make the model
    classifier = RuralIndiaModel(input_size=768, n_topics=config.topics, hidden_size=config.hidden_size).to(device)

    # Make the loss and optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    return classifier, train_loader, validation_loader, test_loader, criterion, optimizer

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        classifier, train_loader, validation_loader, test_loader, criterion, optimizer = make(config)
        train(classifier, train_loader, validation_loader, criterion, optimizer, config.epochs)

    return classifier


if __name__ == '__main__':
    print("Training model...")
    classifier = model_pipeline(config)