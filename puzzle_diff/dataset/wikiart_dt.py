from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

# import tables as tb


class Wikiart_DT(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        all_images = set(list(Path("datasets/wikiart").glob("*/*.jpg")))
        train_file = Path("datasets/data_splits/wikiart_subset_train.txt")
        test_file = Path("datasets/data_splits/wikiart_subset_test.txt")

        with open(train_file, "r") as f:
            train_images_set = set([x.rstrip() for x in f.readlines()])
        with open(test_file, "r") as f:
            test_images_set = set([x.rstrip() for x in f.readlines()])
        if train:
            self.images = [img for img in all_images if img.name in train_images_set]
        else:
            self.images = [img for img in all_images if img.name in test_images_set]
        self.images = sorted(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return Image.open(self.images[index]), None


# class Wikiart_DT_pytables(Dataset):
#   def __init__(self) -> None:
#       super().__init__()

#       # Open the existing HDF5 file
#       file = tb.open_file("wikiart_tr.h5", mode="r", cache_size=32 * 768 * 768 * 3)

#       self.data = file.root.images.images

#   def __len__(self):
#       return self.data.shape[0]

#   def __getitem__(self, index):
#       return Image.fromarray(self.data[index]), None
