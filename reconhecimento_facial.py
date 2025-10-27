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

  faces, ids, labels = carregar_imagens('/contents/faces/')

  print(f"Carregado {len(faces)} imagens de treino.")
  print("Labels:", labels)

  #####
  #####  TREINAR O RECONHECEDOR
  #####
  recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2,
                                                  neighbors=8,
                                                  grid_x=8,
                                                  grid_y=8)
  recognizer.train(faces, ids)

  ## Carregar classificador Haar Cascade
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                'haarcascade_frontalface_default.xml')

  ## Testar a imagem
  img = cv2.imread('/content/teste.jpg')

  if img is None:
    raise FileNotFoundError("Erro ao carregar a imagem de teste.")

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30))

  faces_detectadas = face_cascade.detectMultiScale(gray,
                                                   scaleFactor=1.1,
                                                   minNeighbors=5)

  for (x, y, w, h) in faces_detectadas:
    roi = gray[y:y+h, x:x+w]
    id_pred, confianca = recognizer.predict(roi)
    print("confianca: ", confianca)
    if confianca < 70:
      nome = labels[id_pred]
      texto = f"Conhece: {nome} ({confianca:.2f})"
      cor = (0, 255, 0)
