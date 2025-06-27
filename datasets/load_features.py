import torch
from torch.utils.data import Dataset


class FeatureLoader(Dataset):
    def __init__(self, root):
        self.features = torch.load(root + 'features.pth', weights_only=True).to(torch.float32)
        self.labels = torch.load(root + 'labels.pth', weights_only=True).to(torch.float32)
        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]