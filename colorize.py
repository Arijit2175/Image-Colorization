import numpy as np
import cv2
from cv2 import dnn

proto_file = "Models/colorization_deploy_v2.prototxt"
model_file = "Models/colorization_release_v2.caffemodel"
pts_file = "Models/pts_in_hull.npy"

net = dnn.readNetFromCaffe(proto_file, model_file)
pts = np.load(pts_file)

img = cv2.imread("Input/image.jpg")
scaled = img.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

