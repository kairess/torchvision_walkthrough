import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import cv2
import numpy as np

print('pytorch', torch.__version__)
print('torchvision', torchvision.__version__)

IMG_SIZE = 480
THRESHOLD = 0.95

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

cap = cv2.VideoCapture('imgs/02.mp4')

ret, img = cap.read()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('imgs/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))

while cap.isOpened():
  ret, img = cap.read()

  img = cv2.resize(img, (IMG_SIZE, int(img.shape[0] * IMG_SIZE / img.shape[1])))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  trf = T.Compose([
    T.ToTensor()
  ])

  input_img = trf(img)
  
  out = model([input_img])[0]

  for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
    score = score.detach().numpy()

    if score < THRESHOLD:
      continue

    box = box.detach().numpy()
    keypoints = keypoints.detach().numpy()[:, :2]

    cv2.rectangle(img, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), thickness=2, color=(0, 0, 255))

    for k in keypoints:
      cv2.circle(img, center=tuple(k.astype(int)), radius=2, color=(255, 0, 0), thickness=-1)

    cv2.polylines(img, pts=[keypoints[5:10:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, pts=[keypoints[6:11:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, pts=[keypoints[11:16:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, pts=[keypoints[12:17:2].astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  out_video.write(img)

  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('q'):
    break

out_video.release()
cap.release()
