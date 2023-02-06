import torch
import os
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import cv2
import numpy as np


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df_dataset, cfg, dataset_path, transform=None) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.cfg = cfg
        self.do_resize = cfg["data"].get("do_resize", 1.0) > 0.0
        if cfg["data"]["normalization"] == "pm1":
            print("Apply pm1 normalization")
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif cfg["data"]["normalization"] == "imagenet":
            print("Apply imagenet normalization")
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif cfg["data"]["normalization"] == "None":
            print("No normalization to apply")
            self.normalize = None
        self.transform = transform

    def __len__(self):
        return len(self.df_dataset)

    def expand_greyscale_image_channels(self, grey_image_arr):
        grey_image_arr = np.expand_dims(grey_image_arr, -1)
        grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
        return grey_image_arr_3_channel

    def __getitem__(self, idx):
        label = self.df_dataset.iloc[idx]["cancer"]
        patient_id = str(self.df_dataset.iloc[idx]["patient_id"])
        image_id = str(self.df_dataset.iloc[idx]["image_id"])
        img_path = os.path.join(self.dataset_path, patient_id, image_id + '.png')

        if not os.path.exists(img_path):
            print("The image '{}' does not exist ".format(img_path))
            image = np.zeros((512, 512, 3))
        else:
            image = cv2.imread(img_path)
        # image = self.expand_greyscale_image_channels(image)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.do_resize:
            image = cv2.resize(image, (self.cfg["data"]["size"], self.cfg["data"]["size"]))
        image = transforms.ToTensor()(image)
        if self.normalize is not None:
            image = self.normalize(image)

        return image.float(), label


def create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, cfg, dataset_path, batch_size):
    # 1 - istanzio la classe dataset di train e test
    classification_dataset_test = None
    classification_dataset_train = ClassificationDataset(df_dataset=df_dataset_train,
                                                         cfg=cfg,
                                                         dataset_path=dataset_path)
    classification_dataset_val = ClassificationDataset(df_dataset=df_dataset_val,
                                                       cfg=cfg,
                                                       dataset_path=dataset_path)
    if df_dataset_test is not None:
        classification_dataset_test = ClassificationDataset(df_dataset=df_dataset_test,
                                                            cfg=cfg,
                                                            dataset_path=dataset_path)

    # 2 - istanzio i dataloader
    classification_dataloader_train = DataLoader(dataset=classification_dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 drop_last=False)

    classification_dataloader_val = DataLoader(dataset=classification_dataset_val,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=False)

    classification_dataloader_test = None
    if classification_dataset_test is not None:
        classification_dataloader_test = DataLoader(dataset=classification_dataset_test,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last=False)

    return classification_dataloader_train, classification_dataloader_val, classification_dataloader_test