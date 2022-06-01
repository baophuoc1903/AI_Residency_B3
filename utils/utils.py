import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import numpy as np


def accuracy_test(y_preds, y_true):
    preds = torch.argmax(y_preds, axis=1)
    confu_ma = torch.tensor(confusion_matrix(y_true, preds))
    return torch.diag((confu_ma * torch.eye(confu_ma.shape[0])) / confu_ma.sum(dim=1)).mean().item()


def show_img(im, figsize=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# Define `compare_top_losses` and `compare_most_confused` for examining the model results later
def compare_top_losses(k, path, interp, labels_df):
    tl_val, tl_idx = interp.top_losses(k)
    classes = interp.data.classes

    for i, idx in enumerate(tl_idx):
        fig = plt.figure(figsize=(10, 8))
        columns = 2
        rows = 1

        # Actual Image
        act_im, cl = interp.data.valid_ds[idx]
        cl = int(cl)
        act_cl = classes[cl]
        act_fn = labels_df.loc[labels_df['class_name'] == act_cl]['filename'].values[0]

        # Predicted Image
        pred_cl = interp.pred_class[idx]
        pred_cl = classes[pred_cl]
        pred_fn = labels_df.loc[labels_df['class_name'] == pred_cl]['filename'].values[0]

        print(f'PREDICTION:{pred_cl}, ACTUAL:{act_cl}')
        print(f'Loss: {tl_val[i]:.2f}')
        # Probability: {probs[i][cl]:.4f}'

        # Add image to the left column
        img_path = 'train/' + pred_fn
        im = plt.imread(path + img_path)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(im)

        # Add image to the right column, need to change the tensor shape (permute) for matplotlib
        perm = act_im.data.permute(1, 2, 0)
        fig.add_subplot(rows, columns, 2)
        plt.imshow(perm)
        plt.show()


def compare_most_confused(most_confused, path, labels_df, num_imgs, rank):
    c1 = most_confused[:][rank][0]
    c2 = most_confused[:][rank][1]
    n_confused = most_confused[:3][0][1]
    print(most_confused[:][rank])

    # set the list of
    f_1 = labels_df.loc[labels_df['class_name'] == c1]['filename'].values
    f_2 = labels_df.loc[labels_df['class_name'] == c2]['filename'].values

    fig = plt.figure(figsize=(10, 8))
    columns = 2
    rows = num_imgs
    for i in range(1, columns * rows + 1, 2):
        # Add image to the left column
        img_path = 'train/' + f_1[i]
        im = plt.imread(path + img_path)
        fig.add_subplot(rows, columns, i)
        plt.imshow(im)

        # Add image to the right column
        img_path = 'train/' + f_2[i]
        im = plt.imread(path + img_path)
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(im)

    plt.show()


# Crop function to crop to annotations
def crop(df, path, i):
    image = plt.imread(path + df['filename'][i])
    x1 = df['bbox_x1'][i]
    y1 = df['bbox_y1'][i]
    h = df['bbox_h'][i]
    w = df['bbox_w'][i]

    if len(image.shape) == 3:
        return image[y1:y1 + h, x1:x1 + w, :]
    else:
        # If there are only 2 channels for the image
        return image[y1:y1 + h, x1:x1 + w]