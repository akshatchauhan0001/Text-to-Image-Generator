import streamlit as st
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import streamlit as st
st.set_page_config(page_title="Text-to-Image Generator", page_icon=":guardsman:", layout="wide")
st.title("Text-to-Image Generator")
authorization_token = "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"
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=authorization_token)
pipe.to(device)
def generate_image(text_prompt):
    with autocast(device):
        image = pipe(text_prompt, guidance_scale=8.5).images[0]
        return image
# User input
text_prompt = st.text_input("Enter your prompt:", "A sunset over the ocean")
# Generate image
if st.button('Generate Image'):
    if text_prompt:
        image = generate_image(text_prompt)
        st.write("### Generated Image")
        st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.write("Please enter a prompt to generate an image.")