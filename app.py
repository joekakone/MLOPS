# coding : utf-8


import os
import argparse
import json

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf


TMP = "tmp.jpg"
CLASSES = {
	0: "Class 1",
	1: "Class 2",
	2: "Class 3",
	3: "Class 4",
	4: "Class 5",
}


def read_image(img_path, size, scale=True):
    image = Image.open(img_path)
    image = image.convert("RGB")
    image = image.resize(size)
    if scale:
    	image = np.array(image) / 255.0

    return image

def retreive_and_save(image, size):
	img_name = secure_filename(image.filename)
	image.save(TMP)

	img = read_image(TMP, (size, size))
	os.remove(TMP)

	return img

def decode_results(img, model, idx_to_class):
	probas = np.array(model(img.reshape((1,)+img.shape))[0])
	result = {
		"class": idx_to_class[(np.argmax(probas))],
		"proba": float(np.max(probas)),
		"probas": [{"classs": idx_to_class[i], "proba": float(probas[i])} for i in range(len(probas))]
	}

	return result

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model_path", required=True,
		help="The path to the .h5 file")
	# ap.add_argument("-c", "--classes", required=True,
	# 	help="The path to json file")
	ap.add_argument("-s", "--size", type=int, required=True,
		help="The input size", default=150)
	args = vars(ap.parse_args())

	# load keras model
	model = tf.keras.models.load_model(args["model_path"])

	# load classes matches
	# with open(args["classes"], "r") as f:
	# 	classes = json.load(f)
	# print(classes)
	classes = CLASSES
	print(classes)

	# Flask App
	app = Flask(__name__)

	@app.route("/")
	def home():
		return render_template("index.html")

	@app.route("/api/predict", methods = ["POST"])
	def predict():
		if request.method == "POST":
		    image = request.files["file"]

		    img = retreive_and_save(image, args["size"])

		    return jsonify({"result" : decode_results(img, model, classes)})
		else:
			return jsonify("No valid request, json missing")


	app.run(debug=True, port=5555)


if __name__ == "__main__":
	main()
