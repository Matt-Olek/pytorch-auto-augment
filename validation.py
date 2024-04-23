import utils 
import torch
from collections import OrderedDict

# ------------------------------ Validation ------------------------------ #

def validate(val_loader, model, criterion):
    losses = utils.AverageMeter()
    scores = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            acc = utils.accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            scores.update(acc.item(), input.size(0))
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])
    return log
