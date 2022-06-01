import os
import pandas as pd


def readlines_txt(filepath, with_label=False):
    data = []
    labels = []
    with open(filepath) as fh:
        lines = fh.read().strip('\n').strip().split('\n')
        for line in lines:
            line = line.strip()
            if with_label:
                image, *label = line.split()
                label = ' '.join(label)
                data.append(image+'.jpg')
                labels.append(label)
            else:
                data.append(line)
    return data, labels


def data_to_csv(data_folder, csv_name):
    trainval_txt = os.path.join(data_folder, 'images_variant_trainval.txt')
    test_txt = os.path.join(data_folder, 'images_variant_test.txt')
    variants_txt = os.path.join(data_folder, 'variants.txt')

    trn_fn, trn_lb = readlines_txt(trainval_txt, with_label=True)
    test_fn, test_lb = readlines_txt(test_txt, with_label=True)
    name_labels = readlines_txt(variants_txt, with_label=False)[0]

    is_test = [0]*len(trn_fn) + [1]*len(test_fn)
    trn_fn.extend(test_fn)
    trn_lb.extend(test_lb)
    labels = [name_labels.index(label) for label in trn_lb]

    df = pd.DataFrame({'filename': trn_fn, 'Classes': trn_lb, 'Labels': labels, 'is_test': is_test})
    if not os.path.exists(csv_name):
        df.to_csv(csv_name, index=False)
    return name_labels

if __name__ == '__main__':
    name_labels = data_to_csv(r"/Users/nguyenbaophuoc/Desktop/Paper_AI_Resident/dataset/fgvc-aircraft-2013b/data",
                              r"/Users/nguyenbaophuoc/Desktop/Paper_AI_Resident/dataset/data.csv")
    print(name_labels)
