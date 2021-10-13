"""
Author: Talip Ucar
Email: ucabtuc@gmail.com
Version: 0.1
Description: Data loader for images and RNA data.

----------------------------------
Credit: Following classes / functions are obtained with no, or little modification from
https://github.com/uhlerlab/cross-modal-autoencoders :

NucleiDataset
RNADataset
ToTensorNormalize()
print_nuclei_names()
test_nuclei_dataset()
test_rna_loader()
----------------------------------
"""

import os
import cv2
from skimage import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Loader(object):
    """
    Author: Talip Ucar
    Email: ucabtuc@gmail.com
    """
    def __init__(self, config, dataset_name, eval_mode=False, kwargs={}):
        """
        :param dict config: Configuration dictionary.
        :param str dataset_name: Name of the dataset to use.
        :param bool eval_mode: Whether the dataset is used for evaluation. False by default.
        :param dict kwargs: Additional parameters if needed.
        """
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Set main results directory using database name. 
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(dataset_name, file_path, eval_mode=eval_mode)
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    def get_dataset(self, dataset_name, file_path, eval_mode=False):
        # Create dictionary for loading functions of datasets
        loader_map = {'NucleiDataset': NucleiDataset, 'RNADataset': RNADataset}
        # Get dataset
        dataset = loader_map[dataset_name]
        # Transformation for training dataset. If we are evaluating the model, use ToTensorNormalize.
        train_transform = ToTensorNormalize()
        # Training and Validation datasets
        train_dataset = dataset(datadir=file_path, mode='train', transform=train_transform)
        # Test dataset
        test_dataset = dataset(datadir=file_path, mode='test')
        # Return
        return train_dataset, test_dataset


class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_tensor = sample['tensor']

        # rescale by maximum and minimum of the image tensor
        minX = image_tensor.min()
        maxX = image_tensor.max()
        image_tensor = (image_tensor - minX) / (maxX - minX)

        # resize the inputs
        # torch image tensor expected for 3D operations is (N, C, D, H, W)
        image_tensor = image_tensor.max(axis=0)
        image_tensor = cv2.resize(image_tensor, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        image_tensor = np.clip(image_tensor, 0, 1)
        return torch.from_numpy(image_tensor).view(1, 64, 64)


class NucleiDataset(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize()):
        self.datadir = datadir
        self.mode = mode
        self.images = self.load_images()
        self.transform = transform
        self.threshold = 0.74

    # Utility function to load images from a HDF5 file
    def load_images(self):
        # load labels
        label_data = pd.read_csv(os.path.join(self.datadir, "ratio.csv"))
        label_data_2 = pd.read_csv(os.path.join(self.datadir, "protein_ratios_full.csv"))
        label_data = label_data.merge(label_data_2, how='inner', on='Label')
        label_dict = {name: (float(ratio), np.abs(int(cl) - 2)) for (name, ratio, cl) in
                      zip(list(label_data['Label']), list(label_data['Cor/RPL']), list(label_data['mycl']))}
        label_dict_2 = {name: np.abs(int(cl) - 2) for (name, cl) in
                        zip(list(label_data_2['Label']), list(label_data_2['mycl']))}
        del label_data
        del label_data_2

        # load images
        images_train = []
        images_test = []

        for f in os.listdir(os.path.join(self.datadir, "images")):
            basename = os.path.splitext(f)[0]
            fname = os.path.join(os.path.join(self.datadir, "images"), f)
            if basename in label_dict.keys():
                images_test.append(
                    {'name': basename, 'label': label_dict[basename][0], 'tensor': np.float32(io.imread(fname)),
                     'binary_label': label_dict[basename][1]})
            else:
                try:
                    images_train.append({'name': basename, 'label': -1, 'tensor': np.float32(io.imread(fname)),
                                         'binary_label': label_dict_2[basename]})
                except Exception as e:
                    pass

        # Balance image data set (Minority class is 10% of the dataset)
        images_train2 = []
        for data in images_train:
            for i in range(9 if data["binary_label"] == 0 else 1):
                images_train2.append({'name': data['name'], 'label': data['label'], 'tensor': data['tensor'],
                                     'binary_label': data['binary_label']})


        images_train = images_train2

        if self.mode == 'train':
            return images_train
        elif self.mode == 'test':
            return images_test
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]

        if self.transform:
            # transform the tensor and the particular z-slice
            image_tensor = self.transform(sample)
            return {'tensor': image_tensor, 'name': sample['name'], 'label': sample['label'],
                    'binary_label': sample['binary_label']}
        return sample


class RNADataset(Dataset):
    def __init__(self, datadir, mode='train', transform=ToTensorNormalize()):
        self.datadir = datadir
        self.rna_data, self.labels = self._load_rna_data()

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        rna_sample = self.rna_data[idx]
        cluster = self.labels[idx]
        coro1a = rna_sample[5849]
        rpl10a = rna_sample[2555]
        return {'tensor': torch.from_numpy(rna_sample).float(), 'coro1a': coro1a, 'rpl10a': rpl10a,
                'label': coro1a / rpl10a, 'binary_label': int(cluster)}

    def _load_rna_data(self):
        data = pd.read_csv(os.path.join(self.datadir, "filtered_lognuminorm_pc_rp_7633genes_1396cellsnCD4.csv"),
                           index_col=0)
        data = data.transpose()
        labels = pd.read_csv(os.path.join(self.datadir, "labels_nCD4_corrected.csv"), index_col=0)

        data = labels.merge(data, left_index=True, right_index=True)
        data = data.values

        return data[:, 1:], np.abs(data[:, 0] - 1)


def print_nuclei_names():
    dataset = NucleiDataset(datadir="data/nuclear_crops_all_experiments", mode='test')
    for sample in dataset:
        print(sample['name'])


def test_nuclei_dataset():
    dataset = NucleiDataset(datadir="data/nuclear_crops_all_experiments", mode='train')
    print(len(dataset))
    sample = dataset[0]
    print(sample['tensor'].shape)
    print(sample['binary_label'])

    labels = 0
    for sample in dataset:
        labels += sample['binary_label']
    print(labels)


def test_rna_loader():
    dataset = RNADataset(datadir="data/nCD4_gene_exp_matrices")
    print(len(dataset))
    sample = dataset[0]
    print(torch.max(sample['tensor']))
    print(sample['tensor'].shape)
    for k in sample.keys():
        print(k)
        print(sample[k])


if __name__ == "__main__":
    test_nuclei_dataset()


