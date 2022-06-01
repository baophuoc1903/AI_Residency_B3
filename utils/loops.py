import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast


def train(net, dataloader, optimizer, loss_fcn=torch.nn.BCEWithLogitsLoss(), device=torch.device('cpu')):
    net = net.train()
    total_loss = 0
    n_samples = 0
    correct = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        for data in tepoch:
            with autocast():
                torch.cuda.empty_cache()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # forward + backward + optimize
                outputs = net(inputs)

                loss = loss_fcn(outputs, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                n_samples += outputs.shape[0]
                total_loss += loss

                with torch.no_grad():
                    _, predicted = torch.max(outputs, dim=1)
                    correct += predicted.eq(torch.argmax(labels, dim=1)).sum()
                tepoch.set_postfix(Train_loss=total_loss.item()/n_samples, Train_accuracy=correct.item()/n_samples)

    mloss = total_loss / n_samples
    accuracy = correct / n_samples
    return mloss, accuracy


def evaluate(net, dataloader, loss_fcn=torch.nn.BCEWithLogitsLoss(), device=torch.device('cpu')):
    net = net.eval()
    total_loss = 0
    n_samples = 0
    correct = 0

    with tqdm(dataloader, unit="batch") as tepoch:
        for data in tepoch:
            with autocast():
                torch.cuda.empty_cache()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                loss = loss_fcn(outputs, labels.float())
                n_samples += outputs.shape[0]
                total_loss += loss
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(torch.argmax(labels, dim=1)).sum()
                tepoch.set_postfix(Val_loss=total_loss.item() / n_samples, Val_accuracy=correct.item() / n_samples)

    mloss = total_loss / n_samples
    accuracy = correct / n_samples
    return mloss, accuracy
