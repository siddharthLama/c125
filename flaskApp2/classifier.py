from cgi import test
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C','D', 'E','F', 'G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)
xTrain,xTest,yTrain,yTest = train_test_split(X,y,random_state = 9, train_size = 7500,test_size= 500)
xTrainScaled = xTrain/255.0
xTestScaled = xTest/255.0

clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xTrainScaled,yTrain)

def getPrediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert("L")
    image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_inverted_scaled = np.clip(image_bw_resize-min_pixel,0,255)
    max_pixel = np.max(image_bw_resize)
    image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
    testSample = np.array(image_bw_resize_inverted_scaled).reshape(1,660)
    testPrediction = clf.predict(testSample)
    return testPrediction[0] 
