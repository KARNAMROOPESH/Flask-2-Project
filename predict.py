import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
import numpy as np

X = np.load('./image.npz')['arr_0']
Y = pd.read_csv('./labels.csv')["labels"]
print(pd.Series(Y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L','M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

numberofclasses = len(classes)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=2500, train_size=7500, random_state=9)

Xtrain = Xtrain/255.0
Xtest = Xtest/255.0
classifier = LogisticRegression(random_state=9, solver='saga', multi_class='multinomial')

def getprediction(img):
    print("predict the digit")
    image = Image.open(img)
    imagebw = image.convert('L')
    imagebwresize = imagebw.resize((28,28),Image.ANTIALIAS)

    pixel = 20
    minpixel = np.percentile(imagebwresize ,pixel)
    imagebwresizescaled = np.clip(imagebwresize-minpixel,0,255)
    maxpixel = np.max(imagebwresize)
    imagebwresizescaled = np.asarray(imagebwresizescaled)/maxpixel
    imagearray = np.array(imagebwresizescaled).reshape(1,784)

    imagepredict = classifier.predict(imagearray)
    print(imagepredict)
    return imagepredict[0]

