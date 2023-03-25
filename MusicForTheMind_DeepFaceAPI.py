#!/usr/bin/env python
# coding: utf-8

# ## **IMPORTING REQUIRED MODULES**

# In[3]:


#get_ipython().system('pip install yoloface')
#get_ipython().system('pip install deepface')
#get_ipython().system('pip install fer')
from deepface import DeepFace #importing deepface
import cv2
from yoloface import face_analysis
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import webbrowser


# ## **CAPTURING IMAGE FROM WEBCAM**

# In[32]:


cam = cv2.VideoCapture(0)
result, image = cam.read()
cam.release()
cv2.imwrite('webcam_cap.png',image)


# ## **RETRIEVING ROI(REGION OF INTEREST)**

# In[33]:


face=face_analysis()        
img,box,conf=face.face_detection(image_path='webcam_cap.png',model='tiny')
x,y,h,w = box[0]
crop_img = img[y:y+h, x:x+w]
cv2.imwrite('webcam_cap_cropped.png',crop_img)
print("Region Of Interest Extracted...")
print("Detecting Emotion...")


# ## **READING IMAGE AND GENERATING EMOTION**

# In[34]:


try:
    predictions= DeepFace.analyze('webcam_cap.png',actions=['emotion']) # predictions of deepface model
except:
    predictions= DeepFace.analyze('webcam_cap_cropped.png',actions=['emotion']) # predictions of deepface model
print("You seem to be "+predictions[0]['dominant_emotion'])
#classes in deepface : ‘angry’, ‘disgust’, ‘fear’, ‘happy’, ‘sad’, ‘surprise’, and ‘neutral’


# ## PLAYLIST RECOMMENDATIONS

# In[35]:


common_url = 'https://www.youtube.com/watch?v='
angry = 'x9UHAuyipx8&ab_channel=AcousticMusicCollection'
disgust = 'iKzRIweSBLA&list=PL7v1FHGMOadDghZ1m-jEIUnVUsGMT9jbH'
fear = 'UBBHpoW3AKA&list=PLmgutjZvzLyryoakC3VAlDptXy4ChQGC3&ab_channel=T-Series'
happy = 'JGwWNGJdvx8&list=PLAQ7nLSEnhWTEihjeM1I-ToPDJEKfZHZu&ab_channel=EdSheeran'
sad = 'UBBHpoW3AKA&list=PLmgutjZvzLyryoakC3VAlDptXy4ChQGC3&ab_channel=T-Series'
surprise = 'UupJ9yX3_Bg&list=PLp4nDIl7X7Hd-XnuE5mwiaohwcdMXQMF9&ab_channel=TaylorSwift-Topic'


# In[36]:


emotion = predictions[0]['dominant_emotion']
if emotion == 'angry':
    webbrowser.open(common_url+angry)
elif emotion == 'disgust':
    webbrowser.open(common_url+disgust)
elif emotion == 'fear':
    webbrowser.open(common_url+fear)
elif emotion == 'happy':
    webbrowser.open(common_url+happy)
elif emotion == 'sad':
    webbrowser.open(common_url+sad)
elif emotion == 'neutral':
    webbrowser.open(common_url+happy)
else:
    print('EMotion Not Detected click the image again')


# In[ ]:




