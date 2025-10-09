!pip install opencv-contrib-python --quiet

import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

def carregar_imagens(path):
  faces = []
  ids = []
  labels = {}
  id_atual = 0

  for pessoa in os.listdir(path):
    pessoa_path = os.path.join(path, pessoa)
    if not os.path.isdir(pessoa_path):
      continue

      labels[id_atual] = pessoa

      for img_name in os.listdir(pessoa_path):
        img_path = os.path.join(pessoa_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
          continue

          faces.append(img)
          ids.append(id_atual)

        id_atual += 1
  return faces, np.array(ids), labels
