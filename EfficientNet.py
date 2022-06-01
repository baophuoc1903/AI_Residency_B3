import torch
from torch import nn
from torchvision import models
from torchsummary import summary 
from utils import *
import torch.nn.functional as F
import warnings
import timm
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings("ignore")


class EfficientNet_Mish(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, num_classes)

    def forward(self, image):
        feature = self.resnet(image)
        feature = feature.reshape(-1, 2048)
        feature = nn.ReLU()(self.linear1(feature))
        feature = self.linear2(feature)
        return feature


# class Sift_feature(nn.Module):
#
#     def __init__(self, inp_dim=(736, 736), num_classes=100, sift=False):
#         super().__init__()
#         self.model = EfficientNet_Mish(num_classes)
#         self.sift = sift
#         self.inp_dim = inp_dim
#
#     def forward(self, image):
#         feature1 = self.siam(image)
#         if self.sift:
#             feature2 = self.sift_feature(image)
#             final_feature = torch.cat((feature1, feature2), dim=1)
#         else:
#             final_feature = feature1
#         return final_feature
#
#
# class Image_Compare(nn.Module):
#     def __init__(self, cfg_path='./model.cfg', image_size=(224,224), attention=False, net='resnet50'):
#         super().__init__()
#
#         self.feature_extract = Feature_Extraction(inp_dim=image_size, attention=attention, net=net)
#
#         in_filter = self.feature_extract(torch.randn(1, 3, image_size[0], image_size[1]),
#                                          torch.randn(1, 3, image_size[0], image_size[1])).shape[1]
#
#         self.yolo1 = Darknet(cfg_path, image_size, in_filter=in_filter)
#         self.yolo2 = Darknet(cfg_path, image_size, in_filter=in_filter)
#
#
#     def forward(self, img1, img2):
#         feature = self.feature_extract(img1, img2)
#         bbox1, bbox2 = self.yolo1(feature), self.yolo2(feature)
#
#         return bbox1, bbox2

    
if __name__ == '__main__':
    model = EfficientNet_Mish(num_classes=100)
    summary(model, (3, 224, 224), batch_size=16)
