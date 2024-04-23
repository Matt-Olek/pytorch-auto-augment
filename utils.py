import torch
import matplotlib.pyplot as plt
import pandas as pd


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Calculates the accuracy of the model's predictions"""
    with torch.no_grad():
        Azz, pred = output.max(1)  # Obtient l'indice de la classe prédite avec la probabilité la plus élevée
        pred_cpu = pred.cpu()  # Déplace le tenseur sur le CPU
        target_cpu = target.cpu()  # Déplace également les cibles sur le CPU
        correct = pred_cpu.eq(target_cpu)  # Vérifie si les prédictions correspondent aux cibles
        accuracy = correct.float().mean() * 100.0  # Calcule la précision en pourcentage
    return accuracy

def plot_logs(model_name):
    log = pd.read_csv('models/logs/%s-log.csv' %model_name)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Loss and Accuracy History - %s)' %(model_name))
    ax1.plot(log['epoch'], log['loss'], label='Training Loss')
    ax1.plot(log['epoch'], log['val_loss'], label='Validation Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(log['epoch'], log['acc'], label='Training Accuracy')
    ax2.plot(log['epoch'], log['val_acc'], label='Validation Accuracy')
    ax2.set_title('Accuracy History')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.text(0.5, 0.2, 'Best validation accuracy : %.4f' % log['val_acc'].max(), horizontalalignment='center',
             verticalalignment='center', transform=ax2.transAxes)
    ax2.legend()

    plt.subplots_adjust(hspace=0.5)

    plt.savefig('models/graphs/%s-loss_accuracy_history.png' % model_name)
    plt.close()
    print('Plotted loss and accuracy history to models/%s-loss_accuracy_history.png' % model_name)

    # --- Print the best accuracy --- #

    print('Best validation accuracy achieved: %.4f' % log['val_acc'].max())