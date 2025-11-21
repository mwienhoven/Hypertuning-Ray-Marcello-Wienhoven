import scipy.io
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class Flowers102Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # Load labels (1–102 → we make them 0–101)
        self.labels = (
            scipy.io.loadmat(self.data_dir / "imagelabels.mat")["labels"][0] - 1
        )

        # Load split indices
        setid = scipy.io.loadmat(self.data_dir / "setid.mat")
        if split == "train":
            self.indices = setid["trnid"][0] - 1
        elif split == "val":
            self.indices = setid["valid"][0] - 1
        elif split == "test":
            self.indices = setid["tstid"][0] - 1
        else:
            raise ValueError("Split must be train, val, or test")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx] + 1  # filenames start at 1
        label = self.labels[self.indices[idx]]

        img_path = self.data_dir / "jpg" / f"image_{img_idx:05d}.jpg"
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
