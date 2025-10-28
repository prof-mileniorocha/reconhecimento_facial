!pip install opencv-contrib-python --quiet

import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

# ==========================================
# Função para carregar imagens
# ==========================================
def carregar_imagens(path):
  faces = []
  ids = []
  labels = {}
  id_atual = 0

  for pessoa in os.listdir(path):
    pessoa_path = os.path.join(path, pessoa)
    if not os.path.isdir(pessoa_path):
      continue

    # Associa ID à pessoa
    labels[id_atual] = pessoa

    # Lê todas as imagens da pasta
    for img_name in os.listdir(pessoa_path):
      img_path = os.path.join(pessoa_path, img_name)
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
      if img is None:
        continue
      faces.append(img)
      ids.append(id_atual)

    id_atual += 1

  return faces, np.array(ids), labels


# ==========================================
# Carregar as imagens de treino
# ==========================================
faces, ids, labels = carregar_imagens('/content/faces/')

print(f"Carregado {len(faces)} imagens de treino.")
print("Labels:", labels)

if len(faces) == 0:
  print("⚠️ Nenhuma imagem foi carregada. Verifique se as pastas e imagens estão em /content/faces/.")
else:
  # ==========================================
  # Treinar o reconhecedor LBPH
  # ==========================================
  recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
  recognizer.train(faces, ids)

  # ==========================================
  # Carregar classificador Haar
  # ==========================================
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  # ==========================================
  # Testar a imagem
  # ==========================================
  img = cv2.imread('/content/teste.jpg')

  if img is None:
    raise FileNotFoundError("Erro ao carregar a imagem de teste.")

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces_detectadas = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

  for (x, y, w, h) in faces_detectadas:
    roi = gray[y:y+h, x:x+w]
    id_pred, confianca = recognizer.predict(roi)
    print("Confianca:", confianca)

    if confianca < 70:
      nome = labels[id_pred]
      texto = f"Conhece: {nome} ({confianca:.2f})"
      cor = (0, 255, 0)
    else:
      texto = f"Desconhece ({confianca:.2f})"
      cor = (0, 0, 255)

    # Desenhar retângulo e texto
    cv2.rectangle(img, (x, y), (x + w, y + h), cor, 2)
    cv2.putText(img, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

  # Mostrar imagem final
  cv2_imshow(img)
  cv2.waitKey(1)
  cv2.destroyAllWindows()
