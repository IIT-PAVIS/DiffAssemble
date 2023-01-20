import random
from pathlib import Path

import numpy as np
import tables as tb
from PIL import Image
from tqdm import tqdm

images = set(list(Path("datasets/wikiart").glob("*/*.jpg")))
test_file = Path("datasets/puzzlewikiart-test-metadata.txt")
with open(test_file, "r") as f:
    test_images_set = set([x.split(",")[0] for x in f.readlines()])
test_images = list(test_images_set)
random.shuffle(test_images)
resize_mapping = 768

images = images - test_images_set
images = list(images)
random.shuffle(images)
len_tr = len(images)
len_test = len(test_images)

# data_tr = np.memmap(
#     "wikiart_train.dat",
#     dtype="uint8",
#     mode="w+",
#     shape=(20000, resize_mapping, resize_mapping, 3),
# )
# np.save("wikiart_train_shape.npy", data_tr.shape)

# count = 0
# for i in tqdm(images[:20000]):
#     img = Image.open(i)
#     img = img.resize((resize_mapping, resize_mapping))
#     img = np.asarray(img, dtype=np.uint8)
#     data_tr[count] = img
#     count += 1
# data_tr.flush()


file = tb.open_file("wikiart_tr.h5", mode="w")
group = file.create_group("/", "images")

# Define the shape and the dtype of the images
image_shape = (768, 768, 3)
dtype = np.uint8
compression_filter = tb.Filters(complib="blosc", complevel=9)
chunkshape = (32, *image_shape)

# Create a new PyTables dataset
images_dt = file.create_earray(
    group,
    "images",
    atom=tb.UInt8Atom(),
    shape=(0, *image_shape),
    expectedrows=3000,
    # chunkshape=chunkshape,
    filters=compression_filter,
)

# Fill the dataset with a for loop

for i in tqdm(images[:3000]):
    # if i in test_images_set:
    # continue
    img = Image.open(i)
    img = img.resize((resize_mapping, resize_mapping))
    img = np.asarray(img, dtype=np.uint8)
    images_dt.append(img[None])

# Close the HDF5 file
file.close()

a = 1
