import torch
import csv
import os
import time
import argparse
from utils import generate_prompts, create_directories, save_images, save_labels
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline, AutoPipelineForText2Image

models = {
    "stable_diffusion_2_1": "stabilityai/stable-diffusion-2-1",
    "stable_diffusion_1_0": "stabilityai/stable-diffusion-xl-base-1.0",
    "realistic_vision_1_4": "SG161222/Realistic_Vision_V1.4",
    "kandinsky_2_2": "kandinsky-community/kandinsky-2-2-decoder",
    "openjourney_v4": "prompthero/openjourney-v4"
}

distances = {
    "animals": ["close-up", "far away", "medium distance"]
    ,"fruits": ["close-up", "medium distance"]
}

angles = {
    "animals": ["straight on", "slightly to the right", "slightly to the left", "slightly from above", "the left", "the right"]
    ,"fruits": ["straight on", "above", "the left", "the right"]
}

subjects = {
    "animals": ["dog", "cat", "horse", "spider", "butterfly", "chicken", "sheep", "cow", "squirrel", "elephant"]
    ,"fruits": ["apple braeburn", "avocado", "banana", "orange", "pear", "lemon", "lime", "papaya", "pineapple", "mango"]
}

locations = {}

def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=["stable_diffusion_2_1"], help="Model to use for generating images", nargs='*')
    parser.add_argument("-t", "--theme", default="animals", help="Theme to generate images for", nargs='?')
    parser.add_argument("-c", "--count", default=[5000], help="Amount of images to generate", nargs='*')
    parser.add_argument("-nl", "--nolog", help="Disable logging", action="store_true")

    args = parser.parse_args()

    # Check for amount of arguments
    if len(args.model) == 1:
        if len(args.count) > 1:
            print("Number of parameters don't match")
            return
    else:
        if len(args.count) == 1:
            args.count = args.count * len(args.model)
        elif len(args.count) != len(args.model):
            print("Number of parameters don't match")
            return


    # Generate datasets
    for i, model in enumerate(args.model):
        start = time.time()

        prompts, chosen_subjects = generate_prompts(distances[args.theme], angles[args.theme], subjects[args.theme], int(args.count[i]))

        # Create pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(models[model], torch_dtype=torch.float16)
        if model == "stable_diffusion_2_1":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        # Create directory if it doesn't exist
        create_directories(args.theme, model, subjects[args.theme])

        # Generate images
        for j, prompt in enumerate(prompts):
            image = pipe(prompt).images[0]

            # Get around NSFW filter
            while image.getbbox() is None:
                image = pipe(prompt).images[0]

            # Save images
            save_images(image, args.theme, model, chosen_subjects[j])

        end = time.time()

        # Log model used, amount of images generated and time taken
        if not args.nolog:
            with open(f"./logs.txt", "a") as f:
                f.write(f"Model: {model}, Images: {args.count[i]}, Time: {end-start}s\n")

if __name__ == "__main__":
    main()