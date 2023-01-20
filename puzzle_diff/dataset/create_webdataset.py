import tqdm
import webdataset as wds
from torchvision.datasets import CelebA

celebA_tr = CelebA(
    root="./datasets",
    download=True,
    split="test",
)


sink = wds.TarWriter("dest_test.tar")
dataset = celebA_tr
for index, (input_, _) in enumerate(tqdm.tqdm(dataset)):
    sink.write(
        {
            "__key__": "sample%06d" % index,
            "input.jpg": input_,
        }
    )
sink.close()
