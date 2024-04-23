import models
import utils 
import validation
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import data_loader
from auto_augment import TimeSeriesAutoAugment 

# ------------------------------ Seed ------------------------------ #

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ------------------------------ Training ------------------------------ #

def train_model(train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        acc = utils.accuracy(output, target)

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss)
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    if scheduler is not None:
        lr = scheduler._last_lr[0]
    else:
        lr = optimizer.param_groups[0]['lr']
    return log, lr


def main_loop(auto_augment=False, cutout=False,model_name='default', pprint=True, lr=1e-5, epochs=300, batch_size=200,weight_decay=1e-5,plot=True):
    cudnn.benchmark = True
    # data loading code
    if auto_augment:
        transform_train = TimeSeriesAutoAugment()

    else:
        transform_train = lambda x: x

    print('Training model %s ...' %model_name )

    transform_test = lambda x: x

    train_loader, test_loader = data_loader.get_dataloader(batch_size=batch_size, transform_train=transform_train, transform_test=transform_test,model_name=model_name)

    if auto_augment:
        model_name = model_name + '_(AutoAugment)'
    input_shape = train_loader.dataset[0][0].shape
    print('Input shape:', input_shape)
    model = models.Classifier_RESNET(input_shape=input_shape, nb_classes=5).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)
    epochs = 1000
    log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])
    best_acc = 0
    for epoch in tqdm(range(epochs)):
        if pprint:
            print('Epoch [%d/%d]' %(epoch+1, epochs))
        train_log, lr = train_model(train_loader, model, criterion, optimizer, epoch, scheduler)
        val_log = validation.validate(test_loader, model, criterion)

        if pprint:
            print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
                %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))

        tmp = pd.Series([
            epoch,
            lr,
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

        if not tmp.to_frame().T.empty and not log.empty:
            log = pd.concat([log, tmp.to_frame().T])
        elif not tmp.to_frame().T.empty:
            log = tmp.to_frame().T.copy()
        elif not log.empty:
            log = log
        else:
            log = pd.DataFrame(index=[], columns=[
                'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
            ])
        log.to_csv('models/logs/%s-log.csv' %model_name, index=False)

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/weights/%s.pth' %model_name)
            best_acc = val_log['acc']
            if pprint:
                print("=> saved best model")
        scheduler.step(val_log['loss'])

    if plot:
        utils.plot_logs(model_name)
    return log

# ------------------------------ __main__ ------------------------------ #

if __name__ == '__main__':
    lr = 0.0001
    batch_size = 20
    weight_decay = 1e-6
    model_name = 'ECG5000'
    main_loop(auto_augment=True, cutout=False, model_name=model_name, pprint=False, lr=lr, epochs=300, batch_size=batch_size,weight_decay=weight_decay,plot=True)
    main_loop(auto_augment=False, cutout=False, model_name=model_name,pprint=False, lr=lr, epochs=300, batch_size=batch_size,weight_decay=weight_decay,plot=True)

