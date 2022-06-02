import cv2
import matplotlib.pyplot as plt
from dataset import AirCraftDataset
from model import compute_dense_sift
import math


def show_sift_features(image_path, num_sift=16):
    # batch of 16 key points in 16 equally region in image
    imageread = cv2.imread(image_path)
    imageread = cv2.resize(imageread, (299, 299))
    imageread = cv2.cvtColor(imageread, cv2.COLOR_BGR2RGB)
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
    matches = bf.match(img1_desc, img1_desc)
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
    # visualize_batch()
    img1 = cv2.imread(r"./dataset/fgvc-aircraft-2013b/data/images/0056978.jpg")[:-20, :, ::-1]
    img1 = cv2.resize(img1, (299, 299))
    img2 = cv2.imread(r"./dataset/fgvc-aircraft-2013b/data/images/0894380.jpg")[:-20, :, ::-1]
    img2 = cv2.resize(img2, (299, 299))
    # visualize_sift_effect(img1, img2, n_matches=20)
    plt.subplot(121)
    plt.imshow(img1)
    plt.title(f'0056978.jpg - Class: 707-320')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(img2)
    plt.title(f'0894380.jpg - Class: 707-320')
    plt.xticks([])
    plt.yticks([])
    plt.show()

