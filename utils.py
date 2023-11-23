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

def create_directories(subject, model, classes):
    """
    Create directories for the generated images if they don't exist.
    """
    for c in classes:
        os.makedirs(f"./generated_datasets/{subject}/{model}/256/{c}", exist_ok=True)
        os.makedirs(f"./generated_datasets/{subject}/{model}/native/{c}", exist_ok=True)

def save_images(image, subject, model, cls):
    """
    Save the generated images to the respective directories.
    """
    try:
        index256 = sorted([int(a.split(".")[0]) for a in os.listdir(f"./generated_datasets/{subject}/{model}/256/{cls}") if a.split(".")[0].isdigit()])[-1] + 1
    except:
        index256 = 0
    try:
        indexNative = sorted([int(a.split(".")[0]) for a in os.listdir(f"./generated_datasets/{subject}/{model}/native/{cls}") if a.split(".")[0].isdigit()])[-1] + 1
    except:
        indexNative = 0

    image.resize((256,256)).save(f"./generated_datasets/{subject}/{model}/256/{cls}/{index256}.png")
    image.save(f"./generated_datasets/{subject}/{model}/native/{cls}/{indexNative}.png")

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

def max_samples_per_class(dataset, samples_per_class):
    # Count the number of samples for each class
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1

    # Determine the indices to keep for each class
    indices_per_class = {label: [] for label in class_counts.keys()}
    for idx, (_, label) in enumerate(dataset):
        if len(indices_per_class[label]) < samples_per_class:
            indices_per_class[label].append(idx)

    # Flatten the list of indices
    selected_indices = [idx for indices in indices_per_class.values() for idx in indices]

    return selected_indices
