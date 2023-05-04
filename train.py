import torch
import torchvision
import time
import engine as eng
from torch import nn
import easydict
import torchinfo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pathlib import Path
from sklearn import preprocessing
import pandas as pd
import glob
import eda
import model as md

seed = torch.manual_seed(42) #파이토치 시드 고정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device=='cuda':
    torch.cuda.manual_seed(seed)

print(f'Current device is: {device}')
args = easydict.EasyDict()
args.BATCH_SIZE = 32
args.NUM_EPOCHS = 20
args.desired_score = 0.85
#args.path = Path("/content/gdrive/MyDrive/project/Dacon_tile/data/")
args.path = Path("/Users/Shark/Projects/kaggle_iceberg/data")
args.transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

args.transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def prep():
    print('='*50)
    model = md.resnet18()
    args.model_name = model.__class__.__name__
    print(f'Pytorch {args.model_name} loaded with pre-trained parameters.')

    model.to(device)
    
    train_data, validation_data = eng.get_data(args.BATCH_SIZE, args.path, args.transform, args.transform_test)
    print('Data preperation complete.')

    print('='*50)
    return model, train_data, validation_data

def go(model, train_data, validation_data):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    from timeit import default_timer as timer 
    start_time = timer()

    # Train model
    print("Now training model...")
    model_results = eng.train(model=model, 
                        train_dataloader=train_data,
                        test_dataloader=validation_data,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=args.NUM_EPOCHS, 
                        device=device, 
                        desired_score=args.desired_score)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    return model, model_results


if __name__ == '__main__':
    model, train_data, validation_data = prep()
    model, results = go(model, train_data, validation_data)
    print('Saving model...')
    torch.save(model.state_dict(), args.path / f'models/{args.model_name}.pth')
    print('Model saved!')
    print('Run complete.')
    print('='*50)


