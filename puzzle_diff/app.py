import base64
import io
from io import BytesIO
from typing import List, Tuple

import einops
import gradio as gr
import lightning as L
import numpy as np
import requests
import skimage
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torchvision
import torchvision.transforms as transforms
from lightning.app.components.serve import Image, Number, PythonServer, ServeGradio
from model import spatial_diffusion as sd
from PIL import Image as PILImage
from PIL.Image import Resampling
from torch import Tensor


def encode(image) -> str:

    # convert image to bytes
    with BytesIO() as output_bytes:
        PIL_image = PILImage.fromarray(skimage.img_as_ubyte(image))
        PIL_image.save(output_bytes, "JPEG")  # Note JPG is not a vaild type here
        bytes_data = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return base64_str


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size
) -> List[Tensor]:
    # img2 = einops.rearrange(img, "c h w -> h w c")

    # divide images in non-overlapping patches based on patch size
    # output dim -> a
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    # print(patch_per_dim)

    return xy, patches


class LitGradio(ServeGradio):

    inputs = gr.inputs.Image(type="pil", label="Upload Image for Puzzle")
    outputs = [
        gr.outputs.Image(type="pil", label="Puzzle of input image"),
        gr.outputs.Image(type="pil", label="Output Image"),
    ]

    demo_img = "https://imageio.forbes.com/specials-images/imageserve/60abf319b47a409ca17f4a3f/Pedestrians-cross-Broadway--in-the-SoHo-neighborhood-in-New-York--United-States--May/960x0.jpg?format=jpg&width=960"
    img = PILImage.open(requests.get(demo_img, stream=True).raw)
    img.save("960x0.jpg")
    examples = [["960x0.jpg"]]

    def __init__(self):
        super().__init__()
        self.ready = False
        self._patch_size = 32
        self._transforms = transforms.Compose([transforms.ToTensor()])

    def predict(self, image):
        patch_per_dim = [12, 12]
        graph_in = self.puzzlize(image, patch_per_dim)
        # graph_in = graph_in.to(self._device)
        prediction = self.model.prediction_step(graph_in, 0)

        input_image = self.create_image_from_patches(
            graph_in.patches, prediction[0], n_patches=patch_per_dim
        ).resize((256, 256))

        pred_res = self.create_image_from_patches(
            graph_in.patches, prediction[-1], n_patches=patch_per_dim
        ).resize((256, 256))

        return input_image, pred_res

    def build_model(self):
        ckpt_path = "epoch=124-step=213750.ckpt"
        model = sd.GNN_Diffusion.load_from_checkpoint(ckpt_path)
        model.noise_weight = 1
        self.ready = True
        return model

    def create_image_from_patches(self, patches, pos, n_patches):
        patch_size = 32
        height = patch_size * n_patches[0]
        width = patch_size * n_patches[1]
        new_image = PILImage.new("RGB", (width, height))
        for p in range(patches.shape[0]):
            patch = patches[p, :]
            patch = PILImage.fromarray(
                ((patch.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
            )

            x = pos[p, 0] * (1 - 1 / n_patches[0])
            y = pos[p, 1] * (1 - 1 / n_patches[1])
            x_pos = int((x + 1) * width / 2) - patch_size // 2
            y_pos = int((y + 1) * height / 2) - patch_size // 2
            new_image.paste(patch, (x_pos, y_pos))
        return new_image

    def puzzlize(self, img, patch_per_dim):
        height = patch_per_dim[0] * self._patch_size
        width = patch_per_dim[1] * self._patch_size
        img = img.resize((width, height), resample=Resampling.BICUBIC)
        img = self._transforms(img)

        xy, patches = divide_images_into_patches(img, patch_per_dim, self._patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        adj_mat = torch.ones(
            patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
        )
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_gradio = LitGradio()

    def run(self):
        self.lit_gradio.run()

    def configure_layout(self):
        return [{"name": "", "content": self.lit_gradio}]


app = L.LightningApp(RootFlow())
