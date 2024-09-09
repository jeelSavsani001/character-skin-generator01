import PIL
import requests
import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

model_id = "diffusers/sdxl-instructpix2pix-768"
pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("HarshvardhanCn01/Character_Skin_LoRA")
pipe.fuse_lora()
url = "https://i.pinimg.com/736x/45/6e/9e/456e9e11d43bd2529a262a32c4f20605.jpg"
def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image
image = download_image(url)

prompt = """Create a new skin based on the character, but this time the color scheme should be red and black. The helmet should be predominantly black with cute, white cartoonish expressions and pink accents, similar to the original. The character's suit should be a striking red, with black patches on the arms and legs. The boots should match the overall theme, featuring a combination of black and red. Keep the character's compact build and playful, determined expression. Ensure small details, like the backpack with a tiny bug-like creature attached to it, are preserved, with slight adjustments to fit the new color palette."""
images = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=1).images
images[0].save("res.jpg")
