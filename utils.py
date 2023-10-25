import numpy as np
import os
import csv

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
    index = 0

    while len(prompts) < num_prompts:
        distance = np.random.choice(distances)
        angle = np.random.choice(angles)
        subject = subjects[index % len(subjects)]
        # location = random.choice(locations)

        prompt = f"A {distance} picture of a {subject}, looking from {angle}"
        prompts.append(prompt)
        chosen_subjects.append(subject)
        index += 1

    return prompts, chosen_subjects

def create_directories(subject, model):
    """
    Create directories for the generated images if they don't exist.
    """
    os.makedirs(f"./generated_datasets/{subject}/{model}/256", exist_ok=True)
    os.makedirs(f"./generated_datasets/{subject}/{model}/native", exist_ok=True)

def save_images(image, subject, model, index256, indexNative):
    """
    Save the generated images to the respective directories.
    """
    image.resize((256,256)).save(f"./generated_datasets/{subject}/{model}/256/{index256}.png")
    image.save(f"./generated_datasets/{subject}/{model}/native/{indexNative}.png")

def save_labels(theme, model, chosen_subjects, overwrite=False):
    """
    Save the chosen subjects as labels file in csv format.
    """
    if not overwrite:
        with open(f"./generated_datasets/{theme}/{model}/256/labels.csv", "a") as f:
            writer = csv.writer(f)

            # If the file is empty, add the header
            if os.stat(f"./generated_datasets/{theme}/{model}/256/labels.csv").st_size == 0:
                writer.writerow(["label"])

            for subject in chosen_subjects:
                writer.writerow([subject])

        with open(f"./generated_datasets/{theme}/{model}/native/labels.csv", "a") as f:
            writer = csv.writer(f)

            # If the file is empty, add the header
            if os.stat(f"./generated_datasets/{theme}/{model}/native/labels.csv").st_size == 0:
                writer.writerow(["label"])

            for subject in chosen_subjects:
                writer.writerow([subject])

    else:
        with open(f"./generated_datasets/{theme}/{model}/256/labels.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["label"])
            for subject in chosen_subjects:
                writer.writerow([subject])

        with open(f"./generated_datasets/{theme}/{model}/native/labels.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["label"])
            for subject in chosen_subjects:
                writer.writerow([subject])