import numpy as np
import torch
from torch.utils.data import Dataset


class PandoraImage(Dataset):
    def __init__(self, path, imsize=384):
        self.imsize = imsize        
        self.data = np.fromfile(path, dtype=np.float32)
        self.ss = np.where(self.data==-1.22)[0] # Image seperator locations
        

    def __len__(self):
        return len(self.ss)

#     @classmethod
#     def preprocess(cls, pil_img):
#         img_nd = np.array(pil_img)

#         if len(img_nd.shape) == 2:
#             img_nd = np.expand_dims(img_nd, axis=2)

#         # HWC to CHW
#         img_trans = img_nd.transpose((2, 0, 1))
#         return img_trans

    def __getitem__(self, i):
        xMin = self.data[1+self.ss[i]]
        zMin = self.data[2+self.ss[i]]
        if i+1<self.ss.size:
            nextSep = self.ss[i+1]
        else:
            nextSep = None;
            print("nextSep = self.data.size-1;")
            print(self.ss[i])
            print(nextSep)
        points = self.data[3+self.ss[i]:nextSep].reshape((-1,6))
        x = ((points[:,0]-xMin)/0.3).astype(int)
        z = ((points[:,1]-zMin)/0.3).astype(int)
        Esto = points[:,2:]

        frame = np.zeros((1, 4, self.imsize, self.imsize))
        for x1, z1, Esto1 in zip(x, z, Esto):
            frame[:,:,x1,z1]=Esto1

#         img = frame[:,0,:,:]
#         img = img[:,None,:,:] # Adds an axis of size 1
#         mask = frame[:,1:,:,:]

        img = frame[:,0,:,:]
        img /= 0.01 # Normalisation
        img[img>1.0]=1.0
        mask = frame[0,1:,:,:]

        #print("PandoraImage img",np.shape(img))
        #print("PandoraImage mask",np.shape(mask))
        maskSum = np.sum(mask, axis=0)
        #print("PandoraImage maskSum",np.shape(maskSum))
        mask3 = np.argmax(mask, axis=0)
        #print("PandoraImage mask3",np.shape(mask3))

        maskBkgd = np.where(maskSum==0.0)
        mask3[maskBkgd]=3
#         img = self.preprocess(img)
#         mask = self.preprocess(mask)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask3)}