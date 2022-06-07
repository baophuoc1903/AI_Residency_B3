import cv2
import matplotlib.pyplot as plt
from dataset import AirCraftDataset
from model import compute_dense_sift
import math
from utils import accuracy_test
from model import Efficient_Sift, Efficientnet_b3
from fastai.vision import Learner, LabelSmoothingCrossEntropy, accuracy, DatasetType
from utils import Ranger
from pathlib import PosixPath


def show_sift_features(image_path, num_sift=16):
    # batch of 16 key points in 16 equally region in image
    imageread = cv2.imread(image_path)
    imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2RGB)
    imageread = imageread[:-20, :, :]
    imageread = cv2.resize(imageread, (299, 299))
    imagegray = cv2.cvtColor(imageread, cv2.COLOR_RGB2GRAY)

    step_size = int(math.sqrt(imagegray.shape[0] ** 2 / num_sift))
    keypoints = [cv2.KeyPoint(x + step_size / 2, y + step_size / 2, step_size)
                 for y in range(0, imagegray.shape[0] - step_size // 2, step_size)
                 for x in range(0, imagegray.shape[1] - step_size // 2, step_size)]

    # keypoints = features.detect(imagegray, None)
    # drawKeypoints function is used to draw key points
    output_image = cv2.drawKeypoints(imageread, keypoints, imageread, (0, 255, 0),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(output_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def visualize_batch(data_folder=r'./dataset/fgvc-aircraft-2013b/data',
                    csv_file: str = r'./dataset/data.csv',
                    bs=16):
    dataset = AirCraftDataset(data_folder, csv_file)
    _, _, _, data_test = dataset.extract_data()
    data_test.show_batch(rows=int(bs ** 0.5), figsize=(10, 8))
    plt.show()


def visualize_sift_effect(img1, img2, n_matches=100):
    sift = cv2.SIFT_create(n_matches)
    img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    img1_kp, img1_desc = sift.detectAndCompute(img1, None)
    img2_kp, img2_desc = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(img1_desc, img2_desc)
    # Sort the matches in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # draw the top N matches
    match_img = cv2.drawMatches(img1, img1_kp,
                                img2, img2_kp,
                                matches[:n_matches], img1.copy(), flags=0)
    plt.figure(figsize=(12, 6))
    plt.imshow(match_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # dataset = AirCraftDataset()
    # src, data, src_test, data_test = dataset.extract_data()
    #
    # metrics = ['trn_loss', 'val_loss_and_acc']
    # data_test.batch_size = 64
    # effnet_b3 = "efficientnet-b3"
    #
    # model_weight_save = './'
    # # Mish EfficientNet-B3
    # print("Start valid Mish EfficientNet-B3")
    # mish_model = Efficientnet_b3(data, effnet_b3)
    # mish_learn = Learner(data_test,
    #                      model=mish_model,
    #                      wd=1e-3,
    #                      opt_func=Ranger,
    #                      bn_wd=False,
    #                      true_wd=True,
    #                      metrics=[accuracy],
    #                      loss_func=LabelSmoothingCrossEntropy()
    #                      ).to_fp16()
    # mish_learn.model_dir = model_weight_save
    # tmp_path = mish_learn.path
    # mish_learn.path = PosixPath(model_weight_save)
    # mish_learn.load('best_model')
    # mish_learn.path = tmp_path
    # preds_mish, y_true = mish_learn.get_preds(ds_type=DatasetType.Valid)
    #
    # # SIFT EfficientNet-B3
    # print("Start valid SIFT EfficientNet-B3")
    # sift_model = Efficient_Sift(data, effnet_b3)
    # sift_learn = Learner(data_test,
    #                      model=sift_model,
    #                      wd=1e-3,
    #                      opt_func=Ranger,
    #                      bn_wd=False,
    #                      true_wd=True,
    #                      metrics=[accuracy],
    #                      loss_func=LabelSmoothingCrossEntropy()
    #                      ).to_fp16()
    # sift_learn.model_dir = model_weight_save
    # tmp_path = sift_learn.path
    # sift_learn.path = PosixPath(model_weight_save)
    # sift_learn.load('best_model_sift')
    # sift_learn.path = tmp_path
    # preds_sift, y_true = sift_learn.get_preds(ds_type=DatasetType.Valid)
    #
    # final_acc = accuracy_test(preds_sift.clone() + preds_mish.clone(), y_true)
    # print(final_acc)

    # visualize_batch()
    # img1 = cv2.imread(r"./dataset/fgvc-aircraft-2013b/data/images/0951982.jpg")[:-20, :, ::-1]
    # img1 = cv2.resize(img1, (299, 299))
    # img2 = cv2.imread(r"./dataset/fgvc-aircraft-2013b/data/images/0729223.jpg")[:-20, :, ::-1]
    # img2 = cv2.resize(img2, (299, 299))
    # visualize_sift_effect(img1, img2, n_matches=100)
    show_sift_features(r"./dataset/fgvc-aircraft-2013b/data/images/0951982.jpg", 16)
    # plt.subplot(121)
    # plt.imshow(img1)
    # plt.title(f'0056978.jpg - Class: 707-320')
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(122)
    # plt.imshow(img2)
    # plt.title(f'0894380.jpg - Class: 707-320')
    # plt.xticks([])
    # plt.yticks([])
    plt.show()

