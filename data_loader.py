import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

def get_dataloader(batch_size=128, transform_train=None, transform_test=None,model_name='ECG200'):
    train = pd.read_csv('data/UCRArchive_2018/'+model_name+'/'+model_name+'_TRAIN.tsv', sep='\t', header=None)
    test = pd.read_csv('data/UCRArchive_2018/'+model_name+'/'+model_name+'_TEST.tsv', sep='\t', header=None)
    def f(x):
        return x -1
    train[0] = train[0].apply(lambda x: f(x))
    test[0] = test[0].apply(lambda x: f(x))
    train_np = train.to_numpy()
    test_np = test.to_numpy()   
    train = train_np.reshape(np.shape(train_np)[0], 1, np.shape(train_np)[1])
    test = test_np.reshape(np.shape(test_np)[0], 1, np.shape(test_np)[1])
    y_train = train[:, 0, 0]
    y_test = test[:, 0, 0]
    X_train = train[:, 0, 1:]
    X_test = test[:, 0, 1:]
    # To tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
    y_train = torch.tensor(y_train, dtype=torch.int64).unsqueeze(1) 
    y_test = torch.tensor(y_test, dtype=torch.int64).unsqueeze(1) 
    
    train_dataset =[]
    augmented_train_dataset = []
    for i in range(len(X_train)):
        train_dataset.append((X_train[i], int(y_train[i].item())))

    test_dataset =[]
    for i in range(len(X_test)):
        test_dataset.append((X_test[i], int(y_test[i].item())))

    if transform_train is not None:
        for x, y in train_dataset:
            augmented_train_dataset.append((transform_train(x), y))
            
    train_dataset = train_dataset + augmented_train_dataset
            

    if transform_test is not None:
        test_dataset = [(transform_test(x), y) for x, y in test_dataset]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
