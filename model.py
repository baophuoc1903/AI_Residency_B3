import cv2
import math
from torch import nn
from torchsummary import summary
from utils import *
import warnings
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_dense_sift(image):
    sift = cv2.SIFT_create(16)
    image = np.array(image, dtype=np.float32).transpose(1, 2, 0)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    step_size = int(math.sqrt(image.shape[0] ** 2 / 16))
    kp = [cv2.KeyPoint(x + step_size / 2, y + step_size / 2, step_size) for y in
          range(0, image.shape[0] - step_size // 2, step_size)
          for x in range(0, image.shape[1] - step_size // 2, step_size)]
    __, sift_desc = sift.compute(image, kp)
    return sift_desc


def dense_sift_feature(data):
    sift = np.zeros((data.shape[0], 16, 128))
    for i in range(data.shape[0]):
        sift[i] = compute_dense_sift(data[i])
    return torch.tensor(sift).reshape(-1, 2048)


class Efficientnet_b3(nn.Module):
    def __init__(self, data, model_name='efficientnet-b3', dropout_rate=0.5, drop_connect_rate=0.2):
        super(Efficientnet_b3, self).__init__()
        self.net = EfficientNet.from_pretrained(model_name, dropout_rate=dropout_rate,
                                                drop_connect_rate=drop_connect_rate)
        self.net._fc = nn.Linear(1536, data.c)

    def forward(self, image):
        x = self.net(image)
        return x


class Efficient_Sift(nn.Module):
    def __init__(self, data, model_name='efficientnet-b3', dropout_rate=0.5, drop_connect_rate=0.3, pretrained=''):
        super(Efficient_Sift, self).__init__()
        self.net = Efficientnet_b3(data, model_name, dropout_rate, drop_connect_rate)
        if pretrained:
            state_dict = torch.load(pretrained)
            self.net.load_state_dict(state_dict['model'])
        self.net.net._fc = nn.Linear(1536, 256)
        self.sift = dense_sift_feature
        self.ln1 = nn.Linear(2048, 256)
        self.classifier = nn.Linear(512, data.c)

    def forward(self, image):
        x = self.net(image)
        if str(device) != 'cpu':
            sift = self.sift(image.cpu()).half().to(device)
        else:
            sift = self.sift(image.cpu()).float().to(device)
        sift = self.ln1(sift)
        feature = torch.cat([x, sift], dim=1)
        feature = torch.nn.Mish()(feature)
        feature = self.classifier(feature)
        return feature

    
if __name__ == '__main__':
    from dataset import AirCraftDataset
    dataset = AirCraftDataset()
    src, data, src_test, data_test = dataset.extract_data()
    model = Efficient_Sift(data)
    summary(model, (3, 299, 299), batch_size=16)
