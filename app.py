import PIL
import requests
from diffusers import StableDiffusionImg2ImgPipeline
import torch
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = pipe.to(cuda)
url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


init_image = download_image(url)
# init_image = Image.open("mountains_image.jpeg").convert("RGB").resize((768, 512))
prompt = "A fantasy landscape, trending on artstation"

images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
images[0].save("fantasy_landscape.png")