from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np
import pandas as pd
import sys

def cropAndBin(x, z, binNum, minHits=3, sectionSize=1/10.0):
  needsCropping = True;
  numHits = np.size(x)
  xMin = x.min()
  xMax = x.max()
  zMin = z.min()
  zMax = z.max()

  xDiff=xMax-xMin
  zDiff=zMax-zMin
  if(numHits>20 and zDiff>8 and xDiff>8):
    while(needsCropping):
      needsCropping = False

      if(np.size(x)>0.9*numHits):
        if(np.sum(x<(xMin+xDiff*sectionSize))<minHits):
          needsCropping=True
          minArg = x.argmin()
          x = np.delete(x, minArg)
          z = np.delete(z, minArg)

        if(np.sum(x>(xMax-xDiff*sectionSize))<minHits):
          needsCropping=True
          minArg = x.argmax()
          x = np.delete(x, minArg)
          z = np.delete(z, minArg)

        if(np.sum(z<(zMin+zDiff*sectionSize))<minHits):
          needsCropping=True
          minArg = z.argmin()
          x = np.delete(x, minArg)
          z = np.delete(z, minArg)

        if(np.sum(z>(zMax-zDiff*sectionSize))<minHits):
          needsCropping=True
          minArg = z.argmax()
          x = np.delete(x, minArg)
          z = np.delete(z, minArg)
        
        xMin = x.min()
        xMax = x.max()
        zMin = z.min()
        zMax = z.max()
        xDiff=xMax-xMin
        zDiff=zMax-zMin
  
  div = max(xMax-xMin, zMax-zMin,1)
  x = np.add(x,-xMin)
  x = np.divide(x, div/binNum)
  x = x.astype(int)

  z = np.add(z,-zMin)
  z = np.divide(z, div/binNum)
  z = z.astype(int)
  return x, z

def createMatrix(x, z, binNum):
  mat = np.bincount(x * binNum +z)
  #mat = np.divide(mat,  mat.max())
  mat[mat>5]=5.0
  mat = np.divide(mat, 5);
  mat.resize((binNum, binNum))
  return mat

binNum = 128
inputStr = sys.argv[1]
print(inputStr)
df = pd.DataFrame(inputStr.split(','))

x = df.values[0::2].reshape((np.size(df.values[0::2]))).astype(int)
z = df.values[1::2].reshape((np.size(df.values[1::2]))).astype(int)
print(x)
print(z)
x, z = cropAndBin(x, z, binNum, 8)
mat = createMatrix(x, z, binNum)
mat = mat.reshape((-1,binNum,binNum,1))

model = tf.keras.models.load_model('LEECNNModelF')

pred = model.predict(mat)
print("Only shower [0] or shower + track [1]? Prediction: " + str(pred[0,0]))


