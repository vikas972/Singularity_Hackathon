import numpy as np
import cv2
from PIL import Image
import datetime
from threading import Thread

import time
import pandas as pd

##### Tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

### Deepface by Mani
from deepface import DeepFace #importing deepface
from yoloface import face_analysis
import matplotlib.pyplot as plt
import webbrowser

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)
## For Video Links
common_url = 'https://www.youtube.com/watch?v='
angry = 'x9UHAuyipx8&ab_channel=AcousticMusicCollection'
disgust = 'iKzRIweSBLA&list=PL7v1FHGMOadDghZ1m-jEIUnVUsGMT9jbH'
fear = 'UBBHpoW3AKA&list=PLmgutjZvzLyryoakC3VAlDptXy4ChQGC3&ab_channel=T-Series'
happy = 'JGwWNGJdvx8&list=PLAQ7nLSEnhWTEihjeM1I-ToPDJEKfZHZu&ab_channel=EdSheeran'
sad = 'UBBHpoW3AKA&list=PLmgutjZvzLyryoakC3VAlDptXy4ChQGC3&ab_channel=T-Series'
surprise = 'UupJ9yX3_Bg&list=PLp4nDIl7X7Hd-XnuE5mwiaohwcdMXQMF9&ab_channel=TaylorSwift-Topic'


## For Images
happy1 = "https://content.wepik.com/statics/8773130/preview-page0.jpg"
angry1 = 'https://www.verywellmind.com/thmb/Q0bHSXLTZekPcyregL4lnBCfPA0=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/anger-management-strategies-4178870-final-2a4e9e9e33cc4399b2995c69a46cf84c.png'
disgust1 = 'https://wf-i-can.co.uk/wp-content/uploads/2020/04/strategy-wheel.jpg'
fear1 = 'https://i.redd.it/d4d0tq6yejg51.jpg'
sad1 = 'https://pbs.twimg.com/media/Dx7xlQmWoAA2jk0?format=jpg&name=small'
surprise1 = 'https://i.pinimg.com/564x/c0/ba/73/c0ba73a8b15126d07990db0a66aac025.jpg'




emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv ",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[0]

''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
		def __init__(self, src=0):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	def get_frame(self):
		global cap1
		global df1
		cap1 = WebcamVideoStream(src=0).start()
		image = cap1.read()  
		image=cv2.resize(image,(600,500))
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		face_rects=face_cascade.detectMultiScale(gray,1.3,5)
		res = "https://youtu.be/2Vv-BfVoq4g"
		res1 = "https://content.wepik.com/statics/8773130/preview-page0.jpg"
		for (x,y,w,h) in face_rects:
			global start_time
			cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
			roi_gray_frame = gray[y:y + h, x:x + w]
			cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
			prediction = emotion_model.predict(cropped_img)
			maxindex = int(np.argmax(prediction))
			show_text[0] = maxindex 
			#print("===========================================",music_dist[show_text[0]],"===========================================")
			text_tod = "You seems "+emotion_dict[maxindex]+". Let's watch video."
			cv2.putText(image, text_tod, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
			emotion = emotion_dict[maxindex]
			if emotion == 'Angry':
				# webbrowser.open(common_url+angry)
				res = common_url+angry
				res1 = angry1
			elif emotion == 'Disgusted':
				# webbrowser.open(common_url+disgust)
				res = common_url+disgust
				res1 = disgust1
			elif emotion == 'Fearful':
				# webbrowser.open(common_url+fear)
				res = common_url+fear
				res1 = fear1
			elif emotion == 'Happy':
				# webbrowser.open(common_url+happy)
				res = common_url+happy
				res1 = happy1
			elif emotion == 'Sad':
				# webbrowser.open(common_url+sad)
				res = common_url+sad
				res1 = sad1
			elif emotion == 'Neutral':
				# webbrowser.open(common_url+happy)
				res = common_url+happy
				res1 = happy1

			elif emotion == 'Surprised':
				res = common_url+surprise
				res1 = surprise1
			else:
				print('EMotion Not Detected click the image again')
			
		global last_frame1
		last_frame1 = image.copy()
		pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
		img = Image.fromarray(last_frame1)
		img = np.array(img)
		ret, jpeg = cv2.imencode('.jpg', img)
		return jpeg.tobytes(),res,res1

# def music_rec():
# 	# print('---------------- Value ------------', music_dist[show_text[0]])
# 	df = pd.read_csv(music_dist[show_text[0]])
# 	df = df[['Name','Album','Artist']]
# 	df = df.head(15)
# 	return df
