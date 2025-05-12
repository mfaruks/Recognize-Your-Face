import os
import cv2
import numpy as np
from mtcnn import MTCNN
from sklearn.svm import SVC
from keras_facenet import FaceNet


class HolyClassifier:
  def __init__(self, path):
    self.main_path = path
    self.path0 = os.path.join(path,'0')
    self.path1 = os.path.join(path,'1')
    self.count0 = len(os.listdir(os.path.join(self.main_path,self.path0)))
    self.count1 = len(os.listdir(os.path.join(self.main_path,self.path1)))
    self.mistake0 = 0
    self.mistake1 = 0
    self.detector = MTCNN()
    self.embedder = FaceNet()
    self.model = SVC(kernel='linear',C=0.1)

  def detect_face(self,path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    try:
      x,y,w,h = self.detector.detect_faces(img)[0]['box']
      x,y = abs(x),abs(y)
      face = img[y:y+h,x:x+w]
      return face
    except Exception as e:
      return None

  def get_embeddings(self,faces_arr):
    try:
      embed_vectors = self.embedder.embeddings(faces_arr)
      return embed_vectors
    except Exception as e:
      raise e

  def preprocess_images(self,path):
    images = []
    for filename in os.listdir(path):
      file_path = os.path.join(path,filename)
      img = cv2.imread(file_path)
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      img = cv2.resize(img,(160,160))
      images.append(img)
    return np.stack(images)

  def preprocess_images_end2end(self,path):
    images = []
    count = 0
    for filename in os.listdir(path):
      try:
        filepath = os.path.join(path,filename)
        face = self.detect_face(filepath)
        img = cv2.resize(face,(160,160))
        images.append(img)
      except:
        count+=1
        continue
    embeddings = self.get_embeddings(images) 
    return embeddings,count

  def preprocess_image(self,path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(160,160))
    return np.stack([img])

  def start(self):
    x1, self.mistake1 = self.preprocess_images_end2end(self.path1)
    x0, self.mistake0 = self.preprocess_images_end2end(self.path0)
    X = np.concatenate((x1,x0),axis=0)
    Y = np.concatenate((np.ones(self.count1-self.mistake1,dtype='int'),np.zeros(self.count0-self.mistake0,dtype='int')))
   
    self.model.fit(X, Y)
  
  def predict_with_camera(self):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Kamera Görüntüsü', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            x,y,w,h = self.detector.detect_faces(img)[0]['box']
            x,y = abs(x),abs(y)

            face = img[y:y+h,x:x+w]
            face = cv2.resize(face,(160,160))

            embed = self.embedder.embeddings([face])
            pred = self.model.predict(embed)
            print('prediction: ',pred[0])

        elif key == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

  def predict_with_image(self,path):
    img = self.detect_face(path)
    embd = self.get_embeddings([img])
    pred = self.model.predict(embd)
    return pred[0]
