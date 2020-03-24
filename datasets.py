import torch
from torch.utils.data import Dataset
import h5py
import json
import os


def load_vocab(filepath):
    idx2str = []
    with open(filepath, "r") as fh:
        for line in fh:
            idx2str.append(line.strip())

    str2idx = {x: i for i, x in enumerate(idx2str)}
    return idx2str, str2idx


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL','TEST'}

        # Open hdf5 file where images are stored
        self.train_hf = h5py.File(data_folder + '/train100.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.train_classes = self.train_hf["classes"]
        self.train_attributes = self.train_hf["attributes"]
        self.val_hf = h5py.File(data_folder + '/val100.hdf5', 'r')
        self.val_features = self.val_hf['image_features']
        self.val_classes = self.val_hf["classes"]
        self.val_attributes = self.val_hf["attributes"]

        # Captions per image
        self.cpi = 5
        
        # Load encoded captions 
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths 
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
            
        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        
        # The Nth caption corresponds to the (N // captions_per_image)th image
        objdet = self.objdet[i // self.cpi]
        
        # Load bottom up image features
        if objdet[0] == "v":
            img = torch.FloatTensor(self.val_features[objdet[1]])
            cls = torch.LongTensor(self.val_classes[objdet[1]])
            attr = torch.LongTensor(self.val_attributes[objdet[1]])
        else:
            img = torch.FloatTensor(self.train_features[objdet[1]])
            cls = torch.LongTensor(self.train_classes[objdet[1]])
            attr = torch.LongTensor(self.train_attributes[objdet[1]])

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])
        
        if self.split is 'TRAIN':
            return img, caption, caplen, cls, attr
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, cls, attr, all_captions

    def __len__(self):
        return self.dataset_size
