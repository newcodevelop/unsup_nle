# import torch
# from diffusers import DiffusionPipeline
# from PIL import Image
# import os

# pipe = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True
# ).to("cuda")

# prompts = [
#     "realistic futuristic city at sunset",
#     "realistic dragon flying over a castle",
#     "realistic cyberpunk warrior in the rain",
#     "realistic peaceful lake surrounded by mountains",
#     "realistic astronaut riding a horse on Mars",
#     "realistic cat made of galaxies",
#     "realistic fantasy forest with glowing mushrooms",
#     "realistic neon-lit street in Tokyo",
#     "realistic ship sailing through the clouds",
#     "realistic robot painting a landscape",
#     "realistic surreal dreamscape of clocks and clouds",
#     "realistic wizard casting a spell in a dark cave",
#     "realistic waterfall made of stars",
#     "realistic futuristic train speeding through a jungle",
#     "realistic giant panda playing a guitar",
#     "realistic haunted house in the woods"
# ]

# output_dir = "src/generated_images"
# os.makedirs(output_dir, exist_ok=True)

# for i, prompt in enumerate(prompts):
#     print(f"Generating image {i+1}/16: {prompt}")
#     image = pipe(prompt=prompt, num_inference_steps=50, output_type="pil").images[0]
#     image.save(os.path.join(output_dir, f"image_{i+1:02d}.png"))

# print(f"All images saved in '{output_dir}'")


# def generate_image(prompt: str, steps: int = 50):

#     output = pipe(
#         prompt=prompt,
#         num_inference_steps=steps,
#         output_type="pil"
#     )
#     return output.images[0]

# demo = gr.Interface(
#     fn=generate_image,
#     inputs=[
#         gr.Textbox(label="Prompt", placeholder="waterfall in the forest"),
#         gr.Slider(5, 100, value=50, step=5, label="Inference Steps")
#     ],
#     outputs=gr.Image(label="Generated Image"),
#     title="Stable Diffusion XL Demo",
#     description="Enter a prompt."
# )

# if __name__ == "__main__":
#     demo.launch()

import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

prompts = [
    "realistic futuristic city at sunset",
    "realistic dragon flying over a castle",
    "realistic cyberpunk warrior in the rain",
    "realistic peaceful lake surrounded by mountains",
    "realistic astronaut riding a horse on Mars",
    "realistic cat made of galaxies",
    "realistic fantasy forest with glowing mushrooms",
    "realistic neon-lit street in Tokyo",
    "realistic ship sailing through the clouds",
    "realistic robot painting a landscape",
    "realistic surreal dreamscape of clocks and clouds",
    "realistic wizard casting a spell in a dark cave",
    "realistic waterfall made of stars",
    "realistic futuristic train speeding through a jungle",
    "realistic giant panda playing a guitar",
    "realistic haunted house in the woods"
]

output_dir = "src/generated_images"
os.makedirs(output_dir, exist_ok=True)

# Process prompts in batches
batch_size = 16  # Adjust based on your GPU memory
num_batches = (len(prompts) + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(prompts))
    batch_prompts = prompts[start_idx:end_idx]
    
    print(f"Generating batch {batch_idx+1}/{num_batches} with {len(batch_prompts)} prompts")
    
    # Generate images for the entire batch at once
    results = pipe(prompt=batch_prompts, num_inference_steps=50, output_type="pil").images
    
    # Save each image in the batch
    for i, image in enumerate(results):
        prompt_idx = start_idx + i
        image.save(os.path.join(output_dir, f"image_{prompt_idx+1:02d}.png"))
        print(f"Saved image {prompt_idx+1}/16: {prompts[prompt_idx]}")

print(f"All images saved in '{output_dir}'")
