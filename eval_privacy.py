import os

from torch.utils.data import Dataset
from fire import Fire
from PIL import Image
import torchvision.transforms as T

from fls.fls.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fls.fls.metrics.AuthPct import AuthPct
from fls.fls.metrics.CTTest import CTTest

''
def calc_scores(
        train_dir: str = './data/lung_cancer/generated/samples_iter02/random_sample/size_2048/cond_4.0/real5',
        test_dir: str = './data/lung_cancer/generated/samples_iter02/random_sample/size_2048/cond_4.0/test5',
        generated_dir: str = './data/lung_cancer/generated/samples_iter02/random_sample/size_2048/cond_4.0/all',
):

    feature_extractor = InceptionFeatureExtractor(recompute=True, save_path='./data')

    train_ds = FolderDataset(data_path=train_dir,name='train_set')
    test_ds = FolderDataset(data_path=test_dir,name='test_set')
    gen_ds = FolderDataset(data_path=generated_dir,name='gen_set')

    train_feat = feature_extractor.get_features_from_dataset(train_ds)
    test_feat = feature_extractor.get_features_from_dataset(test_ds)
    gen_feat = feature_extractor.get_features_from_dataset(gen_ds)

    authpct = AuthPct().compute_metric(train_feat, test_feat, gen_feat)
    ct = CTTest().compute_metric(train_feat, test_feat, gen_feat)

    print(f"AuthPct: {authpct}")
    print(f"CT: {ct}")

class FolderDataset(Dataset):

    def __init__(self, data_path='./data/folder/', transforms=T.PILToTensor(), name='trainset'):

        self.data_path = data_path
        self.name = name
        self.transforms = transforms

        self.img_dirs = self.get_img_dirs()
        self.labels = self.get_labels()

    def get_img_dirs(self):
        img_dirs = list()
        for img_name in os.listdir(self.data_path):
            img_dirs.append(os.path.join(self.data_path,img_name))
        return img_dirs

    def get_labels(self):
        #replace with your code in case you need to use the true labels of your data
        return [0 for _ in self.img_dirs]

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):

        img = Image.open(self.img_dirs[idx])
        label = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

if __name__=='__main__':
    Fire(calc_scores)