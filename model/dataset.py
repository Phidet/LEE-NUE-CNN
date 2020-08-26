import numpy as np
import torch
from torch.utils.data import Dataset


class PandoraImage(Dataset):
    def __init__(self, path, imsize=384):
        self.imsize = imsize      
        self.data = np.fromfile(path, dtype=np.float32)
        self.fs = np.where(self.data==np.finfo(np.float32).max)[0] # Image seperator locations
        

    def __len__(self):
        return len(self.fs)


    def __getitem__(self, i):
        
        if i+1<np.size(self.fs): nextSep = self.fs[i+1]# - self.fs[i]
        else: nextSep = None#np.size(self.data)-self.fs[i+1]-1;
        
        ss = np.where(self.data[self.fs[i]:nextSep]==-np.finfo(np.float32).max)[0] + self.fs[i]
        ss = np.append(ss, nextSep)

        frame = np.zeros((6, self.imsize, self.imsize)) # one gaps and one hits channel for every view => 6
        mask = np.zeros((3, self.imsize, self.imsize)).astype(int) # three truth channels fro each view => 9
        
        for j in range(3): 
            gaps = self.data[self.fs[i]+j*self.imsize+1:self.fs[i]+(j+1)*self.imsize+1] 
            gaps = np.repeat(gaps, self.imsize, axis=0).reshape((self.imsize, self.imsize))

            #print("YYY0",i, j, np.shape(ss), np.shape(self.fs))
            xMin = self.data[1+ss[j]]
            zMin = self.data[2+ss[j]]
            points = self.data[3+ss[j]:ss[j+1]].reshape((-1,6))

            x = ((points[:,0]-xMin)/0.3).astype(int)
            z = ((points[:,1]-zMin)/0.3).astype(int)
            Esto = points[:,2:]

            maskTemp = np.zeros((3, self.imsize, self.imsize))
            frame[2*j,:,:] = gaps
            for x1, z1, Esto1 in zip(x, z, Esto):
                frame[2*j+1,x1,z1]=Esto1[0]
                maskTemp[:,x1,z1]=Esto1[1:]
            
            maskSum = np.sum(maskTemp, axis=0)
            maskBkgd = np.where(maskSum==0.0)
            #print("maskSum",np.shape(maskSum))
            mask[j,:,:] = np.argmax(maskTemp, axis=0)
            #print("mask[j,:,:]",np.shape(mask[j,:,:]))
            mask[j,:,:][maskBkgd]=3

        #mask[np.where(mask==2)]=0 # Gives Overlay and showers the same truth value
        frame /= 0.015 # Normalisation
        frame[frame>1.0]=1.0


        return {'image_u': torch.from_numpy(frame[:2,:,:]), 'mask_u': torch.from_numpy(mask[0,:,:]), 
                'image_v': torch.from_numpy(frame[2:4,:,:]), 'mask_v': torch.from_numpy(mask[1,:,:]),
                'image_w': torch.from_numpy(frame[4:6,:,:]), 'mask_w': torch.from_numpy(mask[2,:,:])}
