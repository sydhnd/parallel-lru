from torch.utils.data import Dataset

class SensorWindowDataset(Dataset):
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.w = size

    def __len__(self):
        return len(self.x) - self.w

    def __getitem__(self, i):
        return self.x[i : i + self.w], self.y[i + self.w]
