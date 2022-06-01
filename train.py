import warnings
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.logger import Logger
from utils.loops import train, evaluate
from EfficientNet import EfficientNet_Mish
from dataset import AirCraftDataset
from torch.utils.data import DataLoader
from utils.checkpoint import *
import argparse
from fastai.layers import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(opt, epochs, batch_size=16, augment=False, image_shape=(224, 224)):
    logger = Logger()
    train_data, val_data, test_data = AirCraftDataset('train', augment=augment, shape=image_shape),\
                                      AirCraftDataset('val', shape=image_shape),\
                                      AirCraftDataset('test', shape=image_shape)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=val_data.collate_fn)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=test_data.collate_fn)
    print("Read data successfully")
    net = EfficientNet_Mish(num_classes=train_data.num_labels)
    net = net.to(device)

    learning_rate = opt.lr
    decay = 0.000484
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=4, verbose=True, min_lr=1e-7)

    best_accuracy = 0
    print("Training on", device)
    for epoch in range(epochs):
        print('Epoch {}:'.format(epoch))
        loss_train, acc_train = train(net, trainloader, optimizer, loss_fcn=torch.nn.CrossEntropyLoss(reduction='sum'), device=device)
        loss_val, acc_val = evaluate(net, valloader, loss_fcn=torch.nn.CrossEntropyLoss(reduction='sum'), device=device)
        logger.add_logs(loss_train, loss_val)

        # Update learning rate
        scheduler.step(acc_val)
        s = ('%20s'*4) % ('Train loss', 'Train accuracy', 'Val loss', 'Val accuracy')
        print(s)
        ret = '%20.3g'*4
        print(ret % (loss_train, acc_train, loss_val, acc_val))
        if epochs % 1 == 0:
            save(net, logger, opt, epoch=epoch, best=False)
        if acc_val > best_accuracy:
            save(net, logger, opt, best=True)
            best_accuracy = acc_val

    # Calculate performance on test set
    print('Test performance:')
    results = evaluate(net, testloader, loss_fcn=torch.nn.BCEWithLogitsLoss())
    s = ('%20s' * 2) % ('Test loss', 'Test accuracy')
    print(s)
    ret = '%20.3g' * 2
    print(ret % (results[0], results[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FGVC-AIRCRAFT Training')
    parser.add_argument('--model', type=str, default='efficientnet-b7', help='CNN architecture')
    parser.add_argument('--model_save_dir', default='./run/exp', type=str, help='save weight')
    parser.add_argument('--bs', default=4, type=int, help='batch_size')
    parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    opt = parser.parse_args()
    run(opt, epochs=opt.epochs, batch_size=opt.bs, augment=True)
