# USAGE
# python split_dataset.py --dataset /path/to/original/dataset --output /path/to/output/directory --training_percentage 90

# import the necessary packages
import random
import shutil
import os
import argparse
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to input directory holding original images")
ap.add_argument("-o", "--output", required=True, help="Path to output directory to copy train and validation splits into")
ap.add_argument("-p", "--percentage", type=int, required=True, help="Training percentage")
args = vars(ap.parse_args())

imagePaths = glob.glob(os.path.sep.join([args["dataset"], "**", "*.jpg"]), recursive=True)
random.seed(42)
random.shuffle(imagePaths)

i = int(len(imagePaths) * args["percentage"] / 100)
trainPaths = imagePaths[:i]
validationPaths = imagePaths[i:]

datasets = [
	("training", trainPaths, os.path.sep.join([args["output"], "training"])),
	("validation", validationPaths, os.path.sep.join([args["output"], "validation"]))
]

for (dType, imagePaths, baseOutput) in datasets:
	print("[INFO] building '{}' split".format(dType))

	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	for inputPath in imagePaths:
		filename = inputPath.split(os.path.sep)[-1]
		label = inputPath.split(os.path.sep)[-2]

		labelPath = os.path.sep.join([baseOutput, label])
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)