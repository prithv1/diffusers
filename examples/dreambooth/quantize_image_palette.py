import os
import sys
import ast

import json

from PIL import Image
import numpy as np
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

def replace_rows_with_closest_rows(a, b):
  """Replaces every row of `a` with the row from `b` that is closest in row Euclidean distance.

  Args:
    a: A NumPy array of shape (m, n).
    b: A NumPy array of shape (k, n).

  Returns:
    A NumPy array of shape (m, n).
  """

  # Calculate the Euclidean distance between each row of `a` and every row of `b`.
#   distances = np.linalg.norm(a - b, axis=1)
  distances = euclidean_distances(a, b)
#   print(distances.shape)

  # For each row of `a`, find the row of `b` that is closest to it in Euclidean distance.
  closest_rows = np.argmin(distances, axis=1)

  # Replace each row of `a` with the corresponding row of `b`.
  a[:] = b[closest_rows, :]

  return a

imgf = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234_e2000/im_0.png"
img = Image.open(imgf)

pf = "/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/city2ade_translate_map.json"
palette_data = json.load(open(pf, "r"))
palette_data = list(palette_data.keys())
palette_data = [list(ast.literal_eval(x)) for x in palette_data]
palette_arr = np.array(palette_data)

img_arr = np.array(img)
img_shape = img_arr.shape
img_arr = img_arr.reshape(-1, 3)

img_arr = replace_rows_with_closest_rows(img_arr, palette_arr)
unique_img_vals = [tuple(x) for x in list(np.unique(img_arr, axis=0))]
# print(unique_img_vals)
img_arr = img_arr.reshape(img_shape)
quantized_img = Image.fromarray(img_arr)
quantized_img.save("test_q.png")

