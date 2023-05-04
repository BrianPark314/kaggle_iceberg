import torch
from torch import nn
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import gc
import glob
import requests
import zipfile
from pathlib import Path
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
 
  def __init__(self, data, target, transform=None):
 
    self.data=torch.tensor(data,dtype=torch.float32).permute((0, 3, 1, 2))
    self.target=torch.tensor(target)
 
  def __len__(self):
    return len(self.target)
   
  def __getitem__(self,idx):
    return self.data[idx],self.target[idx]


def get_scaled_imgs(df):
    imgs = []
    label =[]

    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)

        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        imgs.append(np.dstack((a, b, c)))
    return np.array(imgs)
                    

def get_data(BATCH_SIZE, path, transform, transform_test):
    train = pd.read_json(path / 'train.json')
    test = pd.read_json(path / 'test.json')
    scale_train = get_scaled_imgs(train)
    scale_test = get_scaled_imgs(test)
    target = np.array(train['is_iceberg'])

    x_train,x_validation,y_train,y_validation=train_test_split(scale_train,
                                                   target,
                                                   test_size=0.4,
                                                   random_state=1,
                                                   stratify=target)
    
    train_data = MyDataset(x_train, y_train, transform)
    validation_data = MyDataset(x_validation, y_validation, transform_test)
    #test_data = MyDataset(scale_test, None, transform)

    train_dataloader = DataLoader(train_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        )
    valid_dataloader = DataLoader(validation_data, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True, 
                                        )

    #test_dataloader = DataLoader(test_data, 
    #                                    batch_size=BATCH_SIZE, 
    #                                    shuffle=False, 
    #                                    )

    return train_dataloader, valid_dataloader

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int, device,
          desired_score):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        'train_f1':[],
        "test_loss": [],
        "test_acc": [],
        'test_f1':[]
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        model, train_loss, train_acc, train_f1 = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc, test_f1 = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)
        
        # 4. Print out what's happening
        print('\n'+
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"test_f1: {test_f1:.4f} | "
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_f1"].append(test_f1)
        gc.collect()

        if test_f1 > desired_score:
            print('Desired f1 score reached, early stopping')
            return model, results
    # 6. Return the filled results at the end of the epochs
    return model, results

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc, train_f1 = 0, 0, 0 
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        y_preds = y_pred.argmax(1).detach().cpu().numpy().tolist()
        y_labels = y.detach().cpu().numpy().tolist()

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        #f1_score
        f1 = f1_score(y_labels, y_preds)
        train_f1 += f1.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad(set_to_none=True)

        # 4. Loss backward
        loss.requires_grad_(True)
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        gc.collect()

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    train_f1 = train_f1 / len(dataloader)

    return model, train_loss, train_acc, train_f1

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc, test_f1 = 0, 0, 0 
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            y_preds = test_pred_logits.argmax(1).detach().cpu().numpy().tolist()
            y_labels = y.detach().cpu().numpy().tolist()

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            #f1_score
            f1 = f1_score(y_labels, y_preds)
            test_f1 += f1.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    test_f1 = test_f1 / len(dataloader)

    return test_loss, test_acc, test_f1

def inference(model, test_loader):
    model.eval()
    model.to('cpu')
    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(iter(test_loader)):
            imgs = imgs.to('cpu')
            pred = model(imgs)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    return preds

def submission(preds, path, model_name):
    tests = pd.read_csv(path / 'test.csv',index_col='id')
    list_names = list(tests.index.values)
    df = pd.DataFrame(list(zip(list_names, preds)), columns=['id','label'])
    df.to_csv(path / f'{model_name}.csv', index=False, encoding='utf-8')
    return None

