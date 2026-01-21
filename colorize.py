import numpy as np
import cv2
from cv2 import dnn

proto_file = "Models/colorization_deploy_v2.prototxt"
model_file = "Models/colorization_release_v2.caffemodel"
pts_file = "Models/pts_in_hull.npy"

