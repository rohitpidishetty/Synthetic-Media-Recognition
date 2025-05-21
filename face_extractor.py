# import bz2
# import shutil
# import urllib.request

# url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
# output_file = "mmod_human_face_detector.dat.bz2"
# urllib.request.urlretrieve(url, output_file)
# with bz2.open(output_file, "rb") as f_in, open("mmod_human_face_detector.dat", "wb") as f_out:
#     shutil.copyfileobj(f_in, f_out)
# print("Download and extraction complete.")

import dlib
import cv2
import numpy as np


class dlibCNN:
  def __init__(self, IMG_HEIGHT, IMG_WIDTH, pre_trained_mmod):  
    self.IMG_WIDTH, self.IMG_HEIGHT = IMG_WIDTH, IMG_HEIGHT
    self.detector = dlib.cnn_face_detection_model_v1(pre_trained_mmod)

  def srm_pre_processor(self, video_path):
      frames = []
      cap = cv2.VideoCapture(video_path)
      frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      frames_to_capture = 10 
      interval = max(1, frame_count // frames_to_capture)
      
      for i in range(0, frame_count, interval):
          cap.set(cv2.CAP_PROP_POS_FRAMES, i)
          ret, frame = cap.read()
          if not ret:
              break
          rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          detections = self.detector(rgb_frame, 1)
          if detections:
              d = detections[0].rect
              x, y, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
              x, y = max(0, x), max(0, y)
              x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
              face = frame[y:y2, x:x2]
              if face.size > 0:
                  resized_face = cv2.resize(face, (self.IMG_WIDTH, self.IMG_HEIGHT))
                  frames.append(resized_face)
      
      cap.release()
      return None if len(frames) == 0 else np.array(frames)
