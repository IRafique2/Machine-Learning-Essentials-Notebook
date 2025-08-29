

import torch
from diffusers import DiffusionPipeline
import random
from PIL import Image
import os
from tqdm import tqdm
# Load the Stable Diffusion model
from diffusers import DiffusionPipeline

import shutil
from google.colab import files

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda")  

# Parameters
ethnicities = ['Hispanic', "African","Asian", "Middle Eastern","Caucasian"]
age_groups = {
    'child': (1, 20),     # Age range for child
    'young': (20, 40),    # Age range for young
    'old': (40, 100)      # Age range for old
}
genders = ['female', 'male']
hair_colors = ['black', 'brown', 'blonde', 'red']
num_images_per_ethnicity = 100

# Function to generate the prompt
def generate_prompt(ethnicity, age_group, gender, hair_color):
    age_min, age_max = age_group
    age = random.randint(age_min, age_max)
    prompt = f"Headshot of a {age} year old {ethnicity} {gender} person with {hair_color} hair"
    return prompt

# Create directory to save images
output_dir = "generated_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate images with progress bar
def generate_images():
    for ethnicity in ethnicities:
        ethnicity_folder = os.path.join(output_dir, ethnicity)
        if not os.path.exists(ethnicity_folder):
            os.makedirs(ethnicity_folder)

        # Using tqdm for the progress bar in the loop
        for i in tqdm(range(num_images_per_ethnicity), desc=f"Generating images for {ethnicity} ethnicity", unit="image"):
            # Randomly select age group, gender, and hair color
            age_group_key = random.choice(list(age_groups.keys()))
            age_group = age_groups[age_group_key]
            gender = random.choice(genders)
            hair_color = random.choice(hair_colors)

            # Generate the prompt
            prompt = generate_prompt(ethnicity, age_group, gender, hair_color)

            # Generate the image
            image = pipe(prompt).images[0]

            # Save the image
            image_path = os.path.join(ethnicity_folder, f"{ethnicity}_{i+1}.png")
            image.save(image_path)

# Run the image generation process
generate_images()


# Create a zip archive of the generated_images directory
shutil.make_archive("generated_images", 'zip', "generated_images")

# Provide a download link for the zip file
files.download("generated_images.zip")