import torch
import os
import pandas as pd
import datetime


def save(net, logger, opt, epoch=0, best=False):
    if not best:
        path = os.path.join(opt.model_save_dir, 'epoch_' + str(epoch))
        state_dict = net.state_dict()
    else:
        path = os.path.join(opt.model_save_dir, 'best')
        state_dict = net.state_dict()

    # create a dictionary containing the logger info and model info that will be saved
    checkpoint = {
        'logs': logger.get_logs(),
        'params': state_dict
    }
    # save checkpoint
    if not os.path.exists(path):
        os.makedirs(opt.model_save_dir)
    torch.save(checkpoint, path)


def load(net, hps, epoch=0, best=False):
    if not best:
        path = os.path.join(hps['model_save_dir'], 'epoch_' + str(epoch))
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        logs, params = checkpoint['logs'], checkpoint['params']
    else:
        path = os.path.join(hps['model_save_dir'], 'best')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        logs, params = checkpoint['logs'], checkpoint['params']

    net.load_state_dict(params)
    return net, logs


def save_metrics_to_csv(exp_name, run_count, learn, metrics):
    moment = datetime.datetime.now()
    df = pd.DataFrame()
    for m in metrics:
        name = f'{m}_{exp_name}_run-{run_count}_{moment.year}-{moment.month}_{moment.day}'
        ls = []
        if m == 'val_loss_and_acc':
            acc = []
            for learn_metric in learn.recorder.metrics:
                acc.append(learn_metric[0].item())
            ls = learn.recorder.val_losses
            d = {name: ls, 'acc': acc}
            df = pd.DataFrame(d)
        elif m == 'trn_loss':
            for learn_loss in learn.recorder.losses:
                ls.append(learn_loss.item())
            df = pd.DataFrame(ls)
            df.columns = [name]

        df.to_csv(f'{name}_{m}.csv', index=False)
