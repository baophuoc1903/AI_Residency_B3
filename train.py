import warnings
from utils.checkpoint import *
from fastai.callbacks import *
from fastai.vision import Learner
from fastai.layers import LabelSmoothingCrossEntropy
from model import Efficientnet_b3, Efficient_Sift
from dataset import AirCraftDataset
import argparse
from fastai.vision import annealing_cos, annealing_linear, annealing_exp, CallbackList, accuracy
from fastai.utils.mem import Floats, listify, defaults, Optional, Tuple
from utils import Ranger, save_metrics_to_csv
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def FlatCosAnnealScheduler(learn, lr: float = 4e-3, tot_epochs: int = 1, moms: Floats = (0.95, 0.999),
                           start_pct: float = 0.72, curve='cosine'):
    """Manage FCFit training as found in the ImageNet experiments"""
    n = len(learn.data.train_dl)
    anneal_start = int(n * tot_epochs * start_pct)
    batch_finish = ((n * tot_epochs) - anneal_start)
    if curve == "cosine":
        curve_type = annealing_cos
    elif curve == "linear":
        curve_type = annealing_linear
    elif curve == "exponential":
        curve_type = annealing_exp
    else:
        raise ValueError(f"annealing type not supported {curve}")

    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr).schedule_hp('mom', moms[0])
    phase1 = TrainingPhase(batch_finish).schedule_hp('lr', lr, anneal=curve_type).schedule_hp('mom', moms[1])
    phases = [phase0, phase1]
    return GeneralScheduler(learn, phases)


def fit_fc(learn: Learner, tot_epochs: int = None, lr: float = defaults.lr,
           moms: Tuple[float, float] = (0.95, 0.85), start_pct: float = 0.72,
           wd: float = None, callbacks: Optional[CallbackList] = None) -> None:
    """Fit a model with Flat Cosine Annealing"""
    max_lr = learn.lr_range(lr)
    callbacks = listify(callbacks)
    callbacks.append(FlatCosAnnealScheduler(learn, lr, moms=moms, start_pct=start_pct, tot_epochs=tot_epochs))
    callbacks.append(SaveModelCallback(learn, monitor='accuracy', mode='max', name='best_model'))
    learn.fit(tot_epochs, max_lr, wd=wd, callbacks=callbacks)


def run(model, data_test, epochs: int = 20, lr: float = 1e-4, batch_size: int = 64,
        exp_name: str = 'Efficient_mish', model_save_dir: str = './',
        start_pct: float = 0.1, wd: float = 1e-3):

    metrics = ['trn_loss', 'val_loss_and_acc']
    data_test.batch_size = batch_size
    # Manually restarted the gpu kernel and changed the run count as weights seemed to be being saved between runs
    run_count = 1
    learn = Learner(data_test, model=model, opt_func=Ranger,
                    wd=1e-3, bn_wd=False, true_wd=True,
                    metrics=[accuracy],
                    loss_func=LabelSmoothingCrossEntropy()
                    )

    if str(device) != 'cpu':
        learn = learn.to_fp16()

    learn.model_dir = model_save_dir
    fit_fc(learn, tot_epochs=epochs, lr=lr, start_pct=start_pct, wd=wd)

    learn.save(f'{exp_name}_run-{run_count}')
    # SAVE METRICS
    save_metrics_to_csv(exp_name, run_count, learn, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FGVC-AIRCRAFT Training')
    parser.add_argument('--model_save_dir', default='./run/exp', type=str, help='save model')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--bs', default=64, type=int, help='batch_size')
    parser.add_argument('--exp_name', default="Efficient_Mish", type=str, help='Name of experiment-model')
    parser.add_argument('--start_pct', default=0.1, type=float, help='percent of epochs to run cosine annual scheduler')
    parser.add_argument('--wd', default=0.001, type=float, help='wd')
    parser.add_argument('--sift', action='store_true', help='Using SIFT or not')
    parser.add_argument('--data_folder', default=r"./dataset/fgvc-aircraft-2013b/data", type=str, help="data folder")
    parser.add_argument('--csv_file', default=r"./dataset/data.csv", type=str, help="path to data in csv form")
    parser.add_argument('--img_shape', default=299, type=int, help="Input image shape")

    opt = parser.parse_args()
    dataset = AirCraftDataset(data_folder=opt.data_folder, csv_file=opt.csv_file, shape=opt.img_shape)
    src, data, src_test, data_test = dataset.extract_data()
    if opt.sift:
        model = Efficient_Sift(data_test)
    else:
        model = Efficientnet_b3(data_test)
    run(model, data_test, epochs=opt.epochs, lr=opt.lr, batch_size=opt.bs,
        exp_name=opt.exp_name, model_save_dir=opt.model_save_dir,
        start_pct=opt.start_pct, wd=opt.wd)
