import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import EfficientNet
import torch
import datetime
from utils import accuracy_test
from fastai.vision import LabelSmoothingCrossEntropy
import torch.nn.functional as F


x = torch.eye(2)*0.9
x[1][1] = 1
x_i = 1 - x
y = torch.arange(2)
print(F.log_softmax(x, dim=-1))
print(F.log_softmax(x, dim=-1).sum(dim=1))
