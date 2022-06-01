import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import EfficientNet
import torch
import datetime
from utils import accuracy_test

device = torch.device('cpu')
print(str(device) == 'cpu')
