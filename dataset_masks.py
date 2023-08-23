import os
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

def import_small_mask_dataset(
    data_path: str = './data/lung_cancer/ds_iter03/',
    num_labels: int = 5,
    use_split: bool = True,
    size: int = 64,
    transform=T.PILToTensor(),
    batch_size: int = 32,
    num_workers: int = 1,
    small=True,
):

    if not small:
        data_path = data_path.replace('ds_iter03','large_masks')

    trainset = SmallMaskDataset(
        data_path=data_path,
        num_labels=num_labels,
        use_split = use_split,
        size=size,
        mode='train',
        transform=transform,
    )
    testset = SmallMaskDataset(
        data_path=data_path,
        num_labels=num_labels,
        use_split = use_split,
        size=size,
        mode='test',
        transform=transform,
    )
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_loader, test_loader

class SmallMaskDataset(Dataset):

    def __init__(self,
                 data_path: str = './data/lung_cancer/ds_iter03/',
                 num_labels: int = 5,
                 use_split = True,
                 df_label_path: str = None,
                 size: int = 64,
                 mode: str = 'train',
                 transform=T.PILToTensor()):

        self.data_path = data_path
        self.num_labels = num_labels
        if use_split:
            split_idx_path = os.path.join(data_path,'split_indices',f'{mode}_{self.num_labels-1}classes.pickle')
            self.split_idx = pd.read_pickle(split_idx_path)

        self.use_split = use_split

        self.df_label_path = df_label_path
        self.uuid2label = self.extract_labels()

        self.mask_path = os.path.join(self.data_path,f'segmasks_{self.num_labels}classes')
        if not os.path.isdir(self.mask_path):
            Path(self.mask_path).mkdir(parents=True, exist_ok=True)
            filter_mask(self.data_path,num_labels=self.num_labels)

        self.masks,self.labels = self.get_masks_and_labels()
        self.size = size
        self.transform = transform

    def extract_labels(self):
        label_path = os.path.join(self.data_path, 'labels')
        if os.path.isdir(label_path):
            uuid2label = pd.read_pickle(label_path + f'/labels.pickle')

        elif self.df_label_path is not None:
            uuid2label = get_label_dict(self.df_label_path)
            Path(label_path).mkdir(parents=True, exist_ok=True)
            pickle_path = f'{label_path}/labels.pickle'
            with open(pickle_path, 'wb') as handle:
                pickle.dump(uuid2label, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            self.df_labels_path = f'{self.data_path}/labels.csv'
            uuid2label = get_label_dict(self.df_labels_path)
            Path(label_path).mkdir(parents=True, exist_ok=True)
            pickle_path = f'{label_path}/labels.pickle'
            with open(pickle_path, 'wb') as handle:
                pickle.dump(uuid2label, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return uuid2label

    def get_masks_and_labels(self):
        labels = list()
        masks = list()

        if self.use_split:
            for key in self.split_idx:
                if key == 0:
                    continue
                list_path = self.split_idx[key]
                for mask_name in list_path:
                    uuid = mask_name.split('.tiff')[0]
                    label_name = self.uuid2label[uuid]
                    label = self.get_label(label_name)
                    mask_path = self.mask_path + f'/{mask_name}_mask.png'
                    masks.append(mask_path)
                    labels.append(label)
        else:
            for mask_name in  os.listdir(self.mask_path):
                mask_path = os.path.join(self.mask_path,mask_name)
                uuid = mask_name.split('.tiff')[0]
                label_name = self.uuid2label[uuid]
                label = self.get_label(label_name)
                masks.append(mask_path)
                labels.append(label)

        return masks, labels

    def get_label(self,label_name):
        if 'adeno' in label_name.lower():
            label = 0
        elif 'squamous' in label_name.lower():
            label = 1
        elif 'plattenepithel' in label_name.lower():
            label = 1
        else:
            assert False, f'{label_name} is not a valid label name.'
        return label

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDataset[{len(self.images)}]"
        for n, x in enumerate(self.images):
            rep += f'\nImg: {x}\t'
            if n > 10:
                rep += '\n...'
                break
        return rep

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):

        # resize mask to (1,self.size,self.size)
        # small areas of labels {1,2,3,4} will be overwritten by label 0
        # the smaller self.size, the more labels will be overwritten
        mask = Image.open(self.masks[idx]).resize((self.size, self.size), Image.NEAREST)
        label = self.labels[idx]

        if self.transform is not None:
            mask = self.transform(mask)
        return mask, label

def filter_mask(data_path,num_lables=5):
    '''
        if self.num_labels 10 we use the following labels:
            0   ->  0 Unknown
            2   ->  1 Alveole
            4   ->  2 Artery
            5   ->  3 Artifacts
            6   ->  4 Carcinoma
            7   ->  5 Cartilage
            9   ->  6 Connections
            8,14->  7 Necrosis
            18  ->  8 Tumor stroma
    other values->  9 Others
        if self.num_labels 5 we use the following labels:
            0   ->  0 Unknown
            7   ->  1 Carcinoma
            9,18->  2 Necrosis
            23  ->  3 Tumor stroma
    other values->  4 Others
    '''

    folder_path = os.path.join(data_path,'images')
    filelist = glob.glob(f'{folder_path}/*mask.png')

    save_folder = os.path.join(data_path,f'segmasks_{num_labels}classes')
    for mask_path in filelist:
        print(f'filter mask at {mask_path}')
        mask_name = mask_path.split('/')[-1]
        mask = np.array(Image.open(mask_path))
        clean_mask = np.zeros_like(mask)
        if num_labels==10:

            #2->1 Alveole: 2
            clean_mask[mask==2]=1
            mask[mask==2] = 0

            #4->2 Artery: 4
            clean_mask[mask == 4] = 2
            mask[mask == 4] = 0

            #5->3 Artifact: 5
            clean_mask[mask == 5] = 3
            mask[mask == 5] = 0

            #7->4 Carcinoma: 7
            clean_mask[mask == 7] = 4
            mask[mask == 7] = 0

            #8->5 Cartilage: 8
            clean_mask[mask == 8] = 5
            mask[mask == 8] = 0

            #10->6 Connective Tissue: 10
            clean_mask[mask == 10] = 6
            mask[mask == 10] = 0

            #9,18->7 Cells undergoing necrosis: 9 and Necrosis: 18
            clean_mask[mask == 9] = 7
            mask[mask == 9] = 0
            clean_mask[mask == 18] = 7
            mask[mask == 18] = 0

            #23->8 Tumor stroma: 23
            clean_mask[mask == 23] = 8
            mask[mask == 23] = 0

            #other values->9 Others
            clean_mask[mask !=0] = 9

        elif num_labels==5:

            #7->1 Carcinoma
            clean_mask[mask == 7] = 1
            mask[mask == 7] = 0

            #9,18->2 Necrosis
            clean_mask[mask == 9] = 2
            mask[mask == 9] = 0
            clean_mask[mask == 18] = 2
            mask[mask == 18] = 0

            #23->3 Tumor stroma
            clean_mask[mask == 23] = 3
            mask[mask == 23] = 0

            #other values->4 Others
            clean_mask[mask !=0] = 4
        else:
            assert False, f'No label specification for {num_labels} classes.'

        save_path = f'{save_folder}/{mask_name}'
        Image.fromarray(clean_mask).save(save_path)

    return

def get_label_dict(df_path):
    df = pd.read_csv(df_path)
    uuid2label = dict()
    for uuid, subtype in zip(df['uuid'], df['subtype']):
        new_key = uuid.split('/')[1]
        if new_key in uuid2label.keys():
            continue
        uuid2label[new_key] = subtype
    return uuid2label