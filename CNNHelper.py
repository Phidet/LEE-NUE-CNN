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

  div = max(binNum*0.3, xMax-xMin, zMax-zMin) # max ensures that we do not divide by more than one wire per pixel
  x = np.add(x,-xMin)
  x = np.divide(x, div/(binNum-1)) # 0 to (binNum-1)=127 is 128 values
  x = x.astype(int)

  z = np.add(z,-zMin)
  z = np.divide(z, div/(binNum-1))
  z = z.astype(int)
  return x, z

def createMatrix(x1, z1, x2, z2, x3, z3, binNum):
  mat1 = np.bincount(x1 * binNum +z1)
  mat2 = np.bincount(x2 * binNum +z2)
  mat3 = np.bincount(x3 * binNum +z3)

  mat1.resize((binNum, binNum))
  mat2.resize((binNum, binNum))
  mat3.resize((binNum, binNum))
  
  mat = np.stack((mat1, mat2, mat3), axis=2)
  mat[mat>5]=5
  mat = np.divide(mat, 5)
  return mat


binNum = 128
inputStr = sys.argv[1]
#print(inputStr)


#df = pd.DataFrame(inputStr.split(','))

data = np.fromstring(inputStr, dtype=float, sep=',')#df.to_numpy()
#data.astype(np.float);
print("##################  DATA")
print(data)
n = np.size(data)//3
x1 = np.trim_zeros(data[:n:2])
z1 = np.trim_zeros(data[1:n:2])
x2 = np.trim_zeros(data[n:2*n:2])
z2 = np.trim_zeros(data[n+1:2*n:2])
x3 = np.trim_zeros(data[2*n::2])
z3 = np.trim_zeros(data[2*n+1::2])

x1, z1 = cropAndBin(x1, z1, binNum, 8)
x2, z2 = cropAndBin(x2, z2, binNum, 8)
x3, z3 = cropAndBin(x3, z3, binNum, 8)
mat = createMatrix(x1, z1, x2, z2, x3, z3, binNum)
mat = mat.reshape((-1,binNum,binNum,1))

pathToModel = '/usera/jpd/Pandora/PandoraPFA/LArContent-v03_15_04/larpandoracontent/MyArea/'
model = tf.keras.models.load_model(pathToModel+'LEECNNModelThree')

pred = model.predict(mat)
print("Only shower [0] or shower + track [1]? Prediction: " + str(pred[0,0]))


