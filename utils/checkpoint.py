import torch
import os


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
