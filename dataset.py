import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import csv    
import yaml

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import random

from PIL import Image

join=os.path.join

root_path='working/dir/'
local_path='data/path'

data_path=join(root_path,local_path)

def set_global_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

set_global_seed(10)


class ComposeState(T.Compose):
    def __init__(self, transforms):
        self.transforms = []
        self.mask_transforms = []

        for t in transforms:
            self.transforms.append(t)

        self.seed = None
        self.retain_state = False

    def __call__(self, x):
        if self.seed is not None:   # retain previous state
            set_global_seed(self.seed)
        if self.retain_state:    # save state for next call
            self.seed = self.seed or torch.seed()
            set_global_seed(self.seed)
        else:
            self.seed = None    # reset / ignore state

        if isinstance(x, (list, tuple)):
            return self.apply_sequence(x)
        else:
            return self.apply_img(x)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def apply_sequence(self, seq):
        self.retain_state=True
        seq = list(map(self, seq))
        self.retain_state=False
        return seq


def identity(x):
    return x

def get_augmentation(name='identity'):
    if name == 'identity':
        augmentation = identity

    return augmentation

class RandomRotate90():  # Note: not the same as T.RandomRotation(90)
    def __call__(self, x):
        x = x.rot90(random.randint(0, 3), dims=(-1, -2))
        return x

    def __repr__(self):
        return self.__class__.__name__


def create_dataset_csv(data_path: str, threshold=0.3):
    # Use a list comprehension to get all files in the directory ending in '.jpg'
    files = [f for f in Path(data_path).iterdir() if f.name.endswith('jpg')]


    imgs,labels=[],[]
    for f in files:
        f_mask=str(f).replace('.jpg', '_mask.png')
        mask=np.array(Image.open(f_mask))
        if np.mean(mask > 0) > threshold:
            imgs.append(f.stem)
            unique_labels = list(map(str, np.int32(np.unique(mask)) ))
            labels.append(' '.join(unique_labels))

    # Write the image names and labels to a CSV file using the csv.writer class
    with open(Path(data_path, "dataset.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(imgs, labels))


def remove_common_elements(train_set, test_set):
    # Convert train_set and test_set to sets to remove duplicates and enable set operations
    unique_train_set = set(train_set)
    unique_test_set = set(test_set)

    # Compute the unique elements in train_set that are not in test_set
    unique_elements = unique_train_set.difference(unique_test_set)

    # Convert the unique sets back to lists for compatibility with the original function signature
    unique_train_list = list(unique_train_set)
    unique_test_list = list(unique_test_set)

    return unique_train_list, unique_test_list


def get_class_counts(data_path: str = data_path):
    # Get a list of all files in the data directory
    files = os.listdir(data_path)

    # Extract all unique pixel values from the masks
    values = []
    for file in files:
        if file.endswith('.png'):
            mask = np.array(Image.open(os.path.join(data_path, file)))
            values += np.unique(mask).tolist()

    # Map class label 9 to 17 and shift all labels >= 9 down by one
    values = [v if v < 9 else v - 1 if v > 9 else 17 for v in values]

    # Count the number of instances of each class
    class_names = get_class_names(data_path) 
    counts = np.zeros(len(class_names))
    for v in values:
        counts[v] += 1

    return counts


def get_class_names(data_path: str = data_path):
    # Open the YAML file
    with open(os.path.join(data_path, 'class_to_int.yml'), 'r') as file:
        # Load the contents of the file
        contents = yaml.safe_load(file)

    # Extract the class names and the number of classes
    class_dict = contents['features']['target__tfrec']['class_to_int']
    class_names = [None] * len(class_dict)
    for name, idx in class_dict.items():
        if idx != 9:
            class_names[idx - (idx > 9)] = name

    return class_names[:-1]


def dataset_to_dict(data_path: str = data_path):
    # Get the list of class names present in the dataset
    class_names = get_class_names(data_path)
    # Get the number of classes
    num_classes = len(class_names)+1
    # Create an empty dictionary for each class, to store the images belonging to that class
    subsets = {i: [] for i in range(num_classes-1)}

    # Read the dataset CSV file
    with open(join(data_path, 'dataset.csv'), 'r') as file:
        # Use a CSV reader to iterate over the rows of the file
        reader = csv.reader(file)
        for row in reader:
            # Extract the image name and label string from the row
            img, labels_str = row[0], row[1]
            # Convert the label string to a list of integers
            labels = [int(l) for l in labels_str.split()]
            # Iterate over the labels of the image
            for label in labels:
                # Fix the label for necrosis, which is labeled as 9 in the original dataset
                if label == 9:
                    label = 18
                # Decrement the label index for labels greater than or equal to 9, 
                # since we removed necrosis
                if label >= 9:
                    label -= 1
                # Add the image to the subset dictionary for the corresponding class
                subsets[label].append(img)
    return subsets


def split_dataset(data_path: str = data_path, train_size: float = 0.9):
    """
    Splits a dataset into training and test sets, with each set containing data for each class.
    :param data_path: the path to the dataset
    :param train_size: the proportion of the data to use for training
    :return: two dictionaries, one containing the training data and the other containing the test data
    """

    # Create a dictionary that maps class names to their corresponding data
    subset_dict = dataset_to_dict(data_path)

    # Determine the number of classes and create a list of subclasses
    classes = get_class_names(data_path)
    num_classes = len(classes)
    subclasses = list(range(num_classes))

    # Create empty dictionaries to hold the training and test data for each subclass
    train_set, test_set = {}, {}
    for i in subclasses:
        train_set[i], test_set[i] = [], []

    # Split the data for each class into training and test sets
    for i in range(num_classes):
        class_set = subset_dict[i]
        class_counts = len(class_set)

        if class_counts>1:
            # Split the data using the specified train/test ratio
            train_index, test_index = train_test_split(
                                            torch.linspace(0, class_counts - 1, class_counts), 
                                            train_size=train_size)
            # Add the training and test data to the corresponding dictionary
            train_set[i] += [class_set[j] for j in train_index.int()]
            test_set[i] += [class_set[j] for j in test_index.int()]

    return train_set, test_set


def add_unconditional(data_path: str, data_dict: str, no_check=False):
    files = [f for f in Path(data_path).iterdir()]

    for f in files:
        if os.path.isdir(f):
            data_dict = add_unconditional(f,data_dict)

    if not no_check:
        active_images=[]
        for key in data_dict:
            active_images+=data_dict[key]
        for f in files:
            if f.name.endswith('jpg') and f not in active_images:
                data_dict[0].append(f.stem)
    else:
        for f in files:
            if f.name.endswith('jpg'):
                data_dict[0].append(f.stem)

    return data_dict

def import_dataset(
        data_path: str = data_path,
        batch_size: int = 32,
        num_workers: int = 0,
        subclasses: list = None,
        cond_drop_prob: float = 0.5,
        threshold: float = 0.,
        force: bool = False,
        transform=None,
        **kwargs
):
    # Generate the dataset CSV file if it does not exist
    if not os.path.exists(join(data_path, "dataset.csv")) or force:
        create_dataset_csv(data_path=data_path, threshold=threshold)

    train_dict, test_dict = split_dataset(data_path, train_size=0.9)

    # Create the train and test datasets
    train_set = DatasetLung(data_path=data_path, data_dict=train_dict, 
                            subclasses=subclasses, cond_drop_prob=cond_drop_prob,
                            transform=transform)
    test_set = DatasetLung(data_path=data_path, data_dict=test_dict, 
                           subclasses=subclasses, cond_drop_prob=1.,
                           transform=transform)

    # Create the train and test data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


class DatasetLung(Dataset):
    def __init__(self,
            data_path: str,
            data_dict: dict,
            subclasses: list = None,
            cond_drop_prob: float = 0.5,
            extra_unknown_data_path: list = ['unlabelled/data/path1','unlabelled/data/path2',...],
            transform = None):

        if subclasses:
            data_dict = self._subclasses(data_dict,subclasses)

        for extra in extra_unknown_data_path:
            data_dict = add_unconditional(data_path=extra, 
                                          data_dict=data_dict, no_check=True)

        N_classes = len(data_dict)

        self.data_path = data_path
        self.extra = extra_unknown_data_path
        self.data_dict = data_dict
        self.subclasses = subclasses
        self.cutoffs = self._cutoffs(subclasses,cond_drop_prob)
        self.N_classes = N_classes
        self.transform = transform

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDataset[{self.__len__()}]"
        for n, in range(self.N_classes):
            rep += f'\nClass {n} has N samples: {len(self.data_dict[n])}\t'
        return rep

    def __len__(self):
        counts=0
        for i in range(len(self.data_dict)):
            counts+=len(self.data_dict[i])
        return counts

    def _subclasses(self, data_dict: dict, subclasses: list):
        not_subclasses = []
        for k in data_dict.keys():
            if k not in subclasses and k != 0:
                not_subclasses += data_dict[k]
        data_dict = {(i+1): data_dict[k] for i, k in enumerate(subclasses)}
        data_dict[len(subclasses)+1] = not_subclasses
        data_dict[0]=[]
        return data_dict

    def _cutoffs(self, subclasses, cond_drop_prob=0.5):
        probs=[cond_drop_prob/(len(subclasses)+1) for n in range(len(subclasses)+1)]
        probs.insert(0,1.-cond_drop_prob)
        return torch.Tensor(probs).cumsum(dim=0)

    def multi_to_single_mask(self, mask):
        mask=(mask*255).int()
        mask=torch.where(mask==9,17,mask)
        mask=torch.where(mask>9,mask-1,mask)
        if self.tmp_index==0:
            mask=torch.zeros_like(mask)
        elif self.tmp_index==len(self.subclasses)+1:
            uniques=torch.unique(mask).int().tolist()
            uniques=[unique for unique in uniques if unique not in self.subclasses]
            if 0 in uniques:
                uniques.remove(0)
            for unique in uniques:
                mask=torch.where(mask==unique, -1, mask)
            mask=torch.where(mask!=-1, len(self.subclasses)+1, 0)
        else:
            mask=torch.where(mask==self.subclasses[self.tmp_index-1], self.tmp_index, 0)
        return mask

    def unbalanced_data(self):
        # generate a random number in [0,1)
        rand_num = torch.rand(1)
        # find the index of the interval that the random number falls into
        index = torch.sum(rand_num >= self.cutoffs)
        self.tmp_index = index
        # map the index to the appropriate tensor value using PyTorch indexing
        oneclass_data = self.data_dict[index.item()]
        # generate a random number in [0,1)
        rand_num = (torch.rand(1)*len(oneclass_data)).int()
        # extract random img from the selected class
        core_path = oneclass_data[rand_num]
        # return img and mask path
        img_path = join(self.data_path, core_path+'.jpg')
        mask_path = join(self.data_path, core_path+'_mask.png')

        if not os.path.exists(img_path):
            for extra in self.extra:
                extra_path = join(extra, core_path+'.jpg')
                if os.path.exists(extra_path):
                    img_path = extra_path

        # load img and mask
        img = Image.open(img_path)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            h,w,c=np.array(img).shape
            mask=np.zeros((h,w,1)) 

        return img,mask


    def __getitem__(self,idx):

        img, mask = self.unbalanced_data()

        if self.transform is not None:
            img,mask = self.transform((img,mask))

        mask = self.multi_to_single_mask(mask)

        return img,mask
