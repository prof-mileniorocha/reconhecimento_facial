!pip install opencv-contrib-python --quiet

import cv2
import os
import numpy as np
from google.colab.patches import cv2_imshow

# -------------------------
# Carregar imagens de treino
# -------------------------
def carregar_imagens(path, tamanho=(200,200)):
    faces = []
    ids = []
    labels = {}
    id_atual = 0

    for pessoa in sorted(os.listdir(path)):
        if pessoa.startswith('.'):
            continue
        pessoa_path = os.path.join(path, pessoa)
        if not os.path.isdir(pessoa_path):
            continue

        labels[id_atual] = pessoa
        for img_name in sorted(os.listdir(pessoa_path)):
            img_path = os.path.join(pessoa_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, tamanho)
            img = cv2.equalizeHist(img)
            faces.append(img)
            ids.append(id_atual)
        id_atual += 1

    return faces, np.array(ids), labels

faces, ids, labels = carregar_imagens("/content/faces/")
print("Carregado", len(faces), "imagens de treino.")
print("Labels:", labels)

# Verificação de consistência
if len(faces) == 0:
    raise RuntimeError("Nenhuma imagem de treino carregada. Verifique /content/faces/")

# -------------------------
# Treinar reconhecedor
# -------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, ids)

# -------------------------
# Detector de faces
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Carregar imagem de teste
# -------------------------
img = cv2.imread("/content/03.jpg")
if img is None:
    raise FileNotFoundError("A imagem /content/05.jpg não foi encontrada!")

orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ajuste de parâmetros do detectMultiScale conforme experimentação
rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
print("DetectMultiScale retornou", len(rects), "retângulos antes de agrupar.")

# Se houver muitos retângulos sobrepostos, agrupar para reduzir duplicatas.
# groupRectangles espera uma lista de rects em formato [[x,y,w,h], ...]
if len(rects) > 0:
    rects_list = rects.tolist()
    # groupRectangles precisa de um array de rects duplicados pra funcionar melhor.
    rects_np = np.array(rects_list, dtype=np.int32)
    # Converte para [x, y, w, h] -> [x1,y1,x2,y2] para NMS opcional se quiser (aqui usamos groupRectangles)
    # groupRectangles aceita vetor de retângulos em formato [x,y,w,h] com um fator
    grouped_rects, weights = cv2.groupRectangles(rects_list + rects_list, groupThreshold=1, eps=0.2)
    if len(grouped_rects) > 0:
        rects_to_use = grouped_rects
        print("Após groupRectangles:", len(rects_to_use), "retângulos.")
    else:
        rects_to_use = rects
else:
    rects_to_use = rects

# Alternativa de NMS (caso queira usar, com bounding boxes [x1,y1,x2,y2])
def non_max_suppression_fast(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    # boxes input [x, y, w, h] -> convert to x1,y1,x2,y2
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        i = last
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[idxs[:-1]]
        idxs = np.delete(idxs, np.concatenate(([len(idxs)-1], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

# Use NMS se houver muitos retângulos (descomente se preferir)
# rects_to_use = non_max_suppression_fast(np.array(rects_to_use), overlapThresh=0.3)

# -------------------------
# Prever e mostrar informações detalhadas
# -------------------------
min_face_size = 50  # ignora detecções muito pequenas (ajuste conforme sua imagem)
for i, (x, y, w, h) in enumerate(rects_to_use):
    if w < min_face_size or h < min_face_size:
        print(f"Ignorando detecção pequena: {(x,y,w,h)}")
        continue

    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200,200))
    roi = cv2.equalizeHist(roi)

    id_pred, confianca = recognizer.predict(roi)

    # Mostrar dados para debug
    nome_previsto = labels.get(id_pred, "Desconhecido_id")
    print(f"Detecção #{i}: box={(x,y,w,h)} -> id_pred={id_pred}, nome={nome_previsto}, confianca={confianca:.4f}")

    # Ajuste de limiar de decisão (experimente 90, 100, 110)
    limiar = 100
    if confianca < limiar:
        texto = f"{nome_previsto} ({confianca:.2f})"
        cor = (0,255,0)
    else:
        texto = f"Desconhecido ({confianca:.2f})"
        cor = (0,0,255)

    cv2.rectangle(orig, (x,y), (x+w, y+h), cor, 2)
    cv2.putText(orig, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

cv2_imshow(orig)
