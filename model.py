import os
import subprocess
import glob

srcdir = 'design_new'  # Directory containing images and box files
destdir = 'trainfiles'  # Directory to store intermediate training files
output_dir = 'trainoutput'  # Directory to store final trained data
output = os.path.join(os.getcwd(), output_dir)

# Removing all previous trained files
try:
    os.remove(os.path.join('tessdata', 'eng.traineddata'))
except FileNotFoundError:
    pass

# List all jpg files in the source directory
jpgs = [f for f in os.listdir(srcdir) if f.endswith('.jpg')]

# Create a list to store paths to box files
boxes = [os.path.join(srcdir, jpg_file.replace('.jpg', '.box')) for jpg_file in jpgs]

# Define the command to extract unicharset
unicharset_cmd = f"unicharset_extractor --output_unicharset {os.path.join(output_dir, 'unicharset')} {' '.join(boxes)}"

error_files = []

# Loop through each image and its corresponding box file
for image, box in zip(jpgs, boxes):
    with open(box, 'r') as f:
        box_data = f.read()

    psm_mode = 13

    subprocess.run(["tesseract", os.path.join(srcdir, image), os.path.join(destdir, image[:-4]), "-l", "eng", "--psm",
                    str(psm_mode), "nobatch", "box.train"], check=True)

# Generate the unicharset file
try:
    subprocess.run(unicharset_cmd, shell=True, check=True)
except subprocess.CalledProcessError:
    error_files.append((image, box))


# Get a list of all TR files in the trainfiles directory
tr_files = glob.glob(os.path.join(destdir, "*.tr"))


# Run mftraining to generate normproto, inttemp, pffmtable files
try:
    subprocess.run(["mftraining", "-F", "font_properties", "-U", os.path.join(output_dir, "unicharset"), "-O",
                    os.path.join(output_dir, "myfont")] + tr_files)
except subprocess.CalledProcessError as e:
    # Handle the error here
    print("Error occurred during mftraining:", e)

try:
    os.rename("inttemp", os.path.join(output_dir, "inttemp"))
    os.rename("pffmtable", os.path.join(output_dir, "pffmtable"))
    os.rename("shapetable", os.path.join(output_dir, "shapetable"))
except FileNotFoundError:
    error_files.append((image, box))

# Run cntraining to generate normproto, inttemp, pffmtable files for classifier
try:
    subprocess.run(["cntraining", "-D", output_dir], check=True)
except subprocess.CalledProcessError:
    error_files.append((image, box))

# Combine trained data into a single traineddata file
os.chdir(output)
os.rename('inttemp', 'eng.inttemp')
os.rename('normproto', 'eng.normproto')
os.rename('pffmtable', 'eng.pffmtable')
os.rename('shapetable', 'eng.shapetable')
os.rename('unicharset', 'eng.unicharset')
os.system(f"combine_tessdata eng.")
