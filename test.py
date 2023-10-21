import torch
import numpy as np
import csv
import os
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline, AutoPipelineForText2Image

def generate_prompts(distances, angles, subjects, num_prompts):
    """
    Generate a list of semi-random prompts using combinations of the given variables.

    Parameters:
    - distances: List of distance values
    - angles: List of angle values
    - locations: List of location values
    - subject: Subject of the prompt
    - num_prompts: Number of prompts to generate

    Returns:
    List of generated prompts and chosen subjects
    """
    prompts = []
    chosen_subjects = []

    while len(prompts) < num_prompts:
        distance = np.random.choice(distances)
        angle = np.random.choice(angles)
        subject = subjects[num_prompts % len(subjects)]
        # location = random.choice(locations)

        prompt = f"A {distance} picture of a {subject}, looking from {angle}"
        prompts.append(prompt)
        chosen_subjects.append(subject)

    return prompts, chosen_subjects

# Example usage:
distances = ["close-up", "far away", "medium distance"]
angles = ["straight on", "slightly to the right", "slightly to the left", "slightly from above", "the left", "the right"]
subjects = ["dog", "cat", "horse", "spider", "butterfly", "chicken", "sheep", "cow", "squirrel", "elephant"]
# locations = ['beach', 'forest', 'east', 'west']

num_prompts_to_generate = 5000
prompts, chosen_subjects = generate_prompts(distances, angles, subjects, num_prompts_to_generate)

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe1 = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe1.scheduler = DPMSolverMultistepScheduler.from_config(pipe1.scheduler.config)
pipe1 = pipe1.to("cuda")

model_id = "SG161222/Realistic_Vision_V1.4"

pipe2 = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
pipe2 = pipe2.to("cuda")

start = time.time()

# Create directory if it doesn't exist
os.makedirs(f"./generated_datasets/animals/stable_diffusion_2_1/256", exist_ok=True)
for i, prompt in enumerate(prompts):
    image = pipe1(prompt).images[0]

    image.resize((256,256)).save(f"./generated_datasets/animals/stable_diffusion_2_1/256/{i}.png")


# Create directory if it doesn't exist
os.makedirs(f"./generated_datasets/animals/realistic_vision_1_4", exist_ok=True)
for i, prompt in enumerate(prompts):
    image = pipe2(prompt).images[0]

    image.resize((256,256)).save(f"./generated_datasets/animals/realistic_vision_1_4/{i}.png")


end = time.time()

# Save chosen subjects as labels file in csv format
with open("./generated_datasets/animals/stable_diffusion_2_1/256/labels.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["label"])
    for subject in chosen_subjects:
        writer.writerow([subject])

# Save chosen subjects as labels file in csv format
with open("./generated_datasets/animals/realistic_vision_1_4/labels.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["label"])
    for subject in chosen_subjects:
        writer.writerow([subject])

print(f"Time taken: {end - start} seconds")