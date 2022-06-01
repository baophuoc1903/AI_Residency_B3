from dataset import data_to_csv
from fastai.vision import cutout, get_transforms, ImageList
from fastai.vision import DatasetType, ResizeMethod, imagenet_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class AirCraftDataset:
    def __init__(self, data_folder: str = r'./dataset/fgvc-aircraft-2013b/data', csv_file: str = r'./dataset/data.csv',
                 shape: int = 299, do_cutout: bool = False, p_cutout: float = 0.75):
        """
        :param data_folder: folder where data files are stored
        :param csv_file: where to create csv file contain all 10000 sample of fgvc-aircraft dataset
        :param shape: shape of input image fed into model -> (shape, shape, 3)
        :param do_cutout: whether cutout image or not
        :param p_cutout: probability of applying the cutout
        """
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.shape = shape
        self.do_cutout = do_cutout
        self.p_cutout = p_cutout
        self.all_labels = data_to_csv(self.data_folder, self.csv_file)
        self.df = pd.read_csv(self.csv_file)

    def __len__(self):
        return self.df.shape[0]

    def extract_data(self, seed: int = 42, label: str = "Classes", split_size: float = 0.2):
        """
        :param seed: seed to split data into train/val
        :param split_size: factor to split train/val (split_size corresponding to val size)
        :param label: columns in dataframe corresponding to name of labels(string)
        :return:
        """

        if self.do_cutout:
            cutout_tfm = cutout(n_holes=(1, 2), length=(100, 100), p=self.p_cutout)
            tfms = get_transforms(p_affine=0.5, xtra_tfms=[cutout_tfm])
        else:
            tfms = get_transforms(p_affine=0.5)

        trn_labels_df = self.df.loc[self.df['is_test'] == 0, ['filename', 'Classes', 'Labels']].copy()

        # Split trainval to new trainval set with valid size = split_size * trainval size
        src = (ImageList.from_df(trn_labels_df, self.data_folder, folder='images', cols='filename')
               .split_by_rand_pct(valid_pct=split_size, seed=seed)
               .label_from_df(cols=label))

        data = (src.transform(tfms,
                              size=self.shape,
                              resize_method=ResizeMethod.SQUISH,
                              padding_mode='reflection')
                .databunch()
                .normalize(imagenet_stats))

        # Split trainval / test
        src_test = (ImageList.from_df(self.df, self.data_folder, folder='images', cols='filename')
                    # the 'is_test' column has values of 1 for the test set
                    .split_from_df(col='is_test')
                    .label_from_df(cols=label))

        data_test = (src_test.transform(tfms,
                                        size=self.shape,
                                        resize_method=ResizeMethod.SQUISH,
                                        padding_mode='reflection')
                     .databunch()
                     .normalize(imagenet_stats))

        return src, data, src_test, data_test

    @property
    def name_labels(self):
        return self.all_labels

    @property
    def num_labels(self):
        return len(self.all_labels)


if __name__ == "__main__":
    dataset = AirCraftDataset(data_folder=r'./fgvc-aircraft-2013b/data', csv_file=r'./data.csv')
    src, data, src_test, data_test = dataset.extract_data(seed=42, label='Classes', split_size=0.2)

    label_name = dataset.name_labels
    bs = np.random.randint(0, len(dataset), 16)
    print(data_test)
    data_test.show_batch(rows=2, figsize=(10, 8))
    plt.show()

