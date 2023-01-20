from pathlib import Path

from torch.utils.data import Dataset
from tqdm import tqdm


class Nips_dt(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()

        data_path = Path("datasets/text/nips/")
        file_csv = data_path / f"{split}.lower"
        assert file_csv.exists(), f"file {file_csv.name} not found in {file_csv.parent}"
        # csv = pd.read_csv(file_csv, delimiter="<eos>", header=None)
        rows = []

        print(f"Loading NIPS {split} file: {str(file_csv)}")
        self.max_cols = 0
        with open(file_csv, "r") as file:
            for line in tqdm(file):
                line = line.rstrip()
                cols = line.split("<eos>")
                rows.append(cols)
                self.max_cols = max(self.max_cols, len(cols))
        self.data = rows

        a = 1

        # all_images = set(
        #     list(Path("datasets/CelebAMask-HQ/CelebA-HQ-img").glob("*.jpg"))
        # )
        # train_file = Path("datasets/data_splits/CelebA-HQ_train.txt")
        # test_file = Path("datasets/data_splits/CelebA-HQ_test.txt")

        # with open(train_file, "r") as f:
        #     train_images_set = set([x.rstrip() for x in f.readlines()])
        # with open(test_file, "r") as f:
        #     test_images_set = set([x.rstrip() for x in f.readlines()])
        # if train:
        #     self.images = [img for img in all_images if img.name in train_images_set]
        # else:
        #     self.images = [img for img in all_images if img.name in test_images_set]
        # self.images = sorted(self.images)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    dt = Nips_dt(split="train")

    # from transformers import BartTokenizer, BartModel

    # tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    # model = BartModel.from_pretrained("facebook/bart-large")
    # model.return_dict = True

    # # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # inputs = tokenizer(dt[0], return_tensors="pt")
    # outputs = model(**inputs)

    # last_hidden_states = outputs.last_hidden_state

    # a = 1
