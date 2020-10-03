###################################################################
# Cerberus based on https://github.com/LArbys/ub_UResNet/blob/master/build_net.py and the examples in the torchvision library
###################################################################

import torch
import torch.nn.functional as F
from unet_parts import *

class Cerberus(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus, self).__init__()

        self.fme = FeatureMapExchange()


        self.re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.do0 = DownX()

                # self.fme first
        self.fme_v0 = ResidualX(BasicBlock, 3*16, 16, 1)
        self.fme_u0 = ResidualX(BasicBlock, 3*16, 16, 1)
        self.fme_w0 = ResidualX(BasicBlock, 3*16, 16, 1)       

        self.re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.do1 = DownX()

        self.re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.do2 = DownX()

        self.re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.do3 = DownX()

        self.re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.do4 = DownX()

        self.re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.do5 = DownX()

        self.re6 = ResidualX(BasicBlock, 512, 1024, depth)

        # self.fme first
        self.fme_u1 = ResidualX(BasicBlock, 3*1024, 1024, 1)
        self.fme_v1 = ResidualX(BasicBlock, 3*1024, 1024, 1)
        self.fme_w1 = ResidualX(BasicBlock, 3*1024, 1024, 1)

        self.up0 = UpX(1024, 1024)
        self.re7 = ResidualX(BasicBlock, 1024, 512, depth)

        self.up1 = UpX(512, 512)
        self.re8 = ResidualX(BasicBlock, 512, 256, depth)

        self.up2 = UpX(256, 256)
        self.re9 = ResidualX(BasicBlock, 256, 128, depth)

        self.up3 = UpX(128, 128)
        self.re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.up4 = UpX(64, 64)
        self.re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.up5 = UpX(32, 32)
        self.re12 = ResidualX(BasicBlock, 32, 16, depth)

        # self.fme first
        self.fme_u2 = ResidualX(BasicBlock, 3*16, 16, 1)
        self.fme_v2 = ResidualX(BasicBlock, 3*16, 16, 1)
        self.fme_w2 = ResidualX(BasicBlock, 3*16, 16, 1)
     

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        

        ############ Down

        u0 = self.re0(u)
        v0 = self.re0(v)
        w0 = self.re0(w)
        uy = self.do0(u0)
        vy = self.do0(v0)
        wy = self.do0(w0)
        
        uy, vy, wy = self.fme(uy, vy, wy)
        uy = self.fme_u0(uy)
        vy = self.fme_v0(vy)
        wy = self.fme_w0(wy)

        u1 = self.re1(uy)
        v1 = self.re1(vy)
        w1 = self.re1(wy)
        uy = self.do1(u1)
        vy = self.do1(v1)
        wy = self.do1(w1)

        u2 = self.re2(uy)
        v2 = self.re2(vy)
        w2 = self.re2(wy)
        uy = self.do2(u2)
        vy = self.do2(v2)
        wy = self.do2(w2)

        u3 = self.re3(uy)
        v3 = self.re3(vy)
        w3 = self.re3(wy)
        uy = self.do3(u3)
        vy = self.do3(v3)
        wy = self.do3(w3)

        u4 = self.re4(uy)
        v4 = self.re4(vy)
        w4 = self.re4(wy)
        uy = self.do4(u4)
        vy = self.do4(v4)
        wy = self.do4(w4)

        u5 = self.re5(uy)
        v5 = self.re5(vy)
        w5 = self.re5(wy)
        uy = self.do5(u5)
        vy = self.do5(v5)
        wy = self.do5(w5)

        uy = self.re6(uy)
        vy = self.re6(vy)
        wy = self.re6(wy)

        uy, vy, wy = self.fme(uy, vy, wy)
        uy = self.fme_u1(uy)
        vy = self.fme_v1(vy)
        wy = self.fme_w1(wy)

        ############ Up 

        uy = self.up0(uy, u5)
        vy = self.up0(vy, v5)
        wy = self.up0(wy, w5)
        uy = self.re7(uy)
        vy = self.re7(vy)
        wy = self.re7(wy)

        uy = self.up1(uy, u4)
        vy = self.up1(vy, v4)
        wy = self.up1(wy, w4)
        uy = self.re8(uy)
        vy = self.re8(vy)
        wy = self.re8(wy)

        uy = self.up2(uy, u3)
        vy = self.up2(vy, v3)
        wy = self.up2(wy, w3)
        uy = self.re9(uy)
        vy = self.re9(vy)
        wy = self.re9(wy)

        uy = self.up3(uy, u2)
        vy = self.up3(vy, v2)
        wy = self.up3(wy, w2)
        uy = self.re10(uy)
        vy = self.re10(vy)
        wy = self.re10(wy)

        uy = self.up4(uy, u1)
        vy = self.up4(vy, v1)
        wy = self.up4(wy, w1)
        uy = self.re11(uy)
        vy = self.re11(vy)
        wy = self.re11(wy)

        uy = self.up5(uy, u0)
        vy = self.up5(vy, v0)
        wy = self.up5(wy, w0)
        uy = self.re12(uy)
        vy = self.re12(vy)
        wy = self.re12(wy)

        uy, vy, wy = self.fme(uy, vy, wy)
        uy = self.fme_u2(uy)
        vy = self.fme_v2(vy)
        wy = self.fme_w2(wy)

        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return uy, vy, wy



##############################################

class Cerberus2(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus2, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()

        self.fme0 = FeatureMapExchange(16, 16)

        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()

        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()

        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()

        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()

        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()

        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)

        self.fme1 = FeatureMapExchange(1024, 1024)

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)

        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)

        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)

        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.fme2 = FeatureMapExchange(32, 32)

        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)

        self.fme3 = FeatureMapExchange(16, 16)

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        
        uy, vy, wy = self.fme0(uy, vy, wy)

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)

        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)

        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)

        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)

        uy, vy, wy = self.fme1(uy, vy, wy)

        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)

        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)

        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)

        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)

        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)

        uy, vy, wy = self.fme2(uy, vy, wy)

        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)

        uy, vy, wy = self.fme3(uy, vy, wy)

        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return uy, vy, wy









##############################################

class Cerberus3(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus3, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()

        self.fme0 = FeatureMapExchange(16, 16)

        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()

        self.fme1 = FeatureMapExchange(32, 32)

        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()

        self.fme2 = FeatureMapExchange(64, 64)

        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()

        self.fme3 = FeatureMapExchange(128, 128)

        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()

        self.fme4 = FeatureMapExchange(256, 256)

        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()

        self.fme5 = FeatureMapExchange(512, 512)

        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)

        self.fme6 = FeatureMapExchange(1024, 1024)

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)

        self.fme7 = FeatureMapExchange(512, 512)

        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)

        self.fme8 = FeatureMapExchange(256, 256)

        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)

        self.fme9 = FeatureMapExchange(128, 128)

        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.fme10 = FeatureMapExchange(64, 64)

        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.fme11 = FeatureMapExchange(32, 32)

        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)

        self.fme12 = FeatureMapExchange(16, 16)

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        
        uy, vy, wy = self.fme0(uy, vy, wy)

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)

        uy, vy, wy = self.fme1(uy, vy, wy)

        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)

        uy, vy, wy = self.fme2(uy, vy, wy)

        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)

        uy, vy, wy = self.fme3(uy, vy, wy)        

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)

        uy, vy, wy = self.fme4(uy, vy, wy)        

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)

        uy, vy, wy = self.fme5(uy, vy, wy)

        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)

        uy, vy, wy = self.fme6(uy, vy, wy)

        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)

        uy, vy, wy = self.fme7(uy, vy, wy)

        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)

        uy, vy, wy = self.fme8(uy, vy, wy)

        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)

        uy, vy, wy = self.fme9(uy, vy, wy)

        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)

        uy, vy, wy = self.fme10(uy, vy, wy)

        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)

        uy, vy, wy = self.fme11(uy, vy, wy)

        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)

        uy, vy, wy = self.fme12(uy, vy, wy)

        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return uy, vy, wy










##############################################

class Cerberus4(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus4, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()

        self.fme0 = FeatureMapExchange(16, 16)

        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()

        self.fme1 = FeatureMapExchange(32, 32)

        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()

        self.fme2 = FeatureMapExchange(64, 64)

        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)


        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.fme10 = FeatureMapExchange(64, 64)

        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.fme11 = FeatureMapExchange(32, 32)

        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)

        self.fme12 = FeatureMapExchange(16, 16)

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        
        uy, vy, wy = self.fme0(uy, vy, wy)

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)

        uy, vy, wy = self.fme1(uy, vy, wy)

        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)

        uy, vy, wy = self.fme2(uy, vy, wy)

        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
   

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
   

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)


        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)

        uy, vy, wy = self.fme10(uy, vy, wy)

        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)

        uy, vy, wy = self.fme11(uy, vy, wy)

        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)

        uy, vy, wy = self.fme12(uy, vy, wy)

        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return uy, vy, wy




##############################################

class Cerberus5(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus5, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.fme0 = FeatureMapExchange(16, 16)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()


        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.fme1 = FeatureMapExchange(32, 32)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()


        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.fme2 = FeatureMapExchange(64, 64)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()


        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)


        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.fme10 = FeatureMapExchange(64, 64)

        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.fme11 = FeatureMapExchange(32, 32)

        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)

        self.fme12 = FeatureMapExchange(16, 16)

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        
        uy, vy, wy = self.fme0(uy, vy, wy)

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)

        uy, vy, wy = self.fme1(uy, vy, wy)

        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)

        uy, vy, wy = self.fme2(uy, vy, wy)

        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
   

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
   

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)


        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)

        uy, vy, wy = self.fme10(uy, vy, wy)

        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)

        uy, vy, wy = self.fme11(uy, vy, wy)

        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)

        uy, vy, wy = self.fme12(uy, vy, wy)

        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return uy, vy, wy





##############################################

class Cerberus6(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus6, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.fme0 = FeatureMapExchange(16, 16)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()


        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.fme1 = FeatureMapExchange(32, 32)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()


        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.fme2 = FeatureMapExchange(64, 64)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()


        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)


        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.fme10 = FeatureMapExchange(64, 64)

        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.fme11 = FeatureMapExchange(32, 32)

        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)

        self.fme12 = FeatureMapExchange(16, 16)

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        u0, v0, w0 = self.fme0(u0, v0, w0)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        u1, v1, w1 = self.fme1(u1, v1, w1)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)


        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        u2, v2, w2 = self.fme2(u2, v2, w2)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)


        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
   

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
   

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)


        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)
        uy, vy, wy = self.fme10(uy, vy, wy)


        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)
        uy, vy, wy = self.fme11(uy, vy, wy)


        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)
        uy, vy, wy = self.fme12(uy, vy, wy)


        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return uy, vy, wy



##############################################

class Cerberus2U(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus2U, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()

        self.fme0 = FeatureMapExchange(16, 16)

        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()

        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()

        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()

        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()

        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()

        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)

        self.fme1 = FeatureMapExchange(1024, 1024)

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)

        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)

        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)

        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)

        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)

        self.fme2 = FeatureMapExchange(32, 32)

        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)

        self.fme3 = FeatureMapExchange(16, 16)

        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, input):
        
        ####### Beginning of U-ResNet #######
        #####################################
        
        u = input[:,:2,:,:]
        v = input[:,2:4,:,:]
        w = input[:,4:6,:,:]

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        
        uy, vy, wy = self.fme0(uy, vy, wy)

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)

        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)

        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)

        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)

        uy, vy, wy = self.fme1(uy, vy, wy)

        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)

        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)

        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)

        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)

        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)

        uy, vy, wy = self.fme2(uy, vy, wy)

        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)

        uy, vy, wy = self.fme3(uy, vy, wy)

        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return torch.cat((uy, vy, wy), dim=1)
        
        
        
        
##############################################

class Cerberus3F(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus3F, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.fme0 = FeatureMapExchange(16, 16)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()


        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.fme1 = FeatureMapExchange(32, 32)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()


        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.fme2 = FeatureMapExchange(64, 64)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()


        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.fme3 = FeatureMapExchange(128, 128)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.fme4 = FeatureMapExchange(256, 256)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.fme5 = FeatureMapExchange(512, 512)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)

        self.fme6 = FeatureMapExchange(1024, 1024)

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.fme7 = FeatureMapExchange(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.fme8 = FeatureMapExchange(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.fme9 = FeatureMapExchange(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.fme10 = FeatureMapExchange(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)


        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.fme11 = FeatureMapExchange(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)


        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.fme12 = FeatureMapExchange(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)


        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        #u = input[:,:2,:,:]
        #v = input[:,2:4,:,:]
        #w = input[:,4:6,:,:]

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        u0, v0, w0 = self.fme0(u0, v0, w0)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        u1, v1, w1 = self.fme1(u1, v1, w1)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)


        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        u2, v2, w2 = self.fme2(u2, v2, w2)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)


        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        u3, v3, w3 = self.fme3(u3, v3, w3)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
        

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        u4, v4, w4 = self.fme4(u4, v4, w4)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
        

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        u5, v5, w5 = self.fme5(u5, v5, w5)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)

        uy, vy, wy = self.fme6(uy, vy, wy)

        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy, vy, wy = self.fme7(uy, vy, wy)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy, vy, wy = self.fme8(uy, vy, wy)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy, vy, wy = self.fme9(uy, vy, wy)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy, vy, wy = self.fme10(uy, vy, wy)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)


        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy, vy, wy = self.fme11(uy, vy, wy)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)


        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy, vy, wy = self.fme12(uy, vy, wy)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)


        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        #return torch.cat((uy, vy, wy), dim=1)
        return uy, vy, wy
        
        
##############################################

class Cerberus3F2(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus3F2, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.fme0 = FeatureMapExchange(16, 16)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()


        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.fme1 = FeatureMapExchange(32, 32)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()


        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.fme2 = FeatureMapExchange(64, 64)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()


        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.fme3 = FeatureMapExchange(128, 128)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.fme4 = FeatureMapExchange(256, 256)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.fme5 = FeatureMapExchange(512, 512)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.fme6 = FeatureMapExchange(1024, 1024)
        

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.fme7 = FeatureMapExchange(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.fme8 = FeatureMapExchange(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.fme9 = FeatureMapExchange(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.fme10 = FeatureMapExchange(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)


        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.fme11 = FeatureMapExchange(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)


        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.fme12 = FeatureMapExchange(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)


        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        #u = input[:,:2,:,:]
        #v = input[:,2:4,:,:]
        #w = input[:,4:6,:,:]

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        u0, v0, w0 = self.fme0(u0, v0, w0)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        u1, v1, w1 = self.fme1(u1, v1, w1)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)


        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        u2, v2, w2 = self.fme2(u2, v2, w2)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)


        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        u3, v3, w3 = self.fme3(u3, v3, w3)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
        

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        u4, v4, w4 = self.fme4(u4, v4, w4)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
        

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        u5, v5, w5 = self.fme5(u5, v5, w5)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)
        uy, vy, wy = self.fme6(uy, vy, wy)


        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy, vy, wy = self.fme7(uy, vy, wy)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy, vy, wy = self.fme8(uy, vy, wy)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy, vy, wy = self.fme9(uy, vy, wy)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy, vy, wy = self.fme10(uy, vy, wy)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)


        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy, vy, wy = self.fme11(uy, vy, wy)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)


        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy, vy, wy = self.fme12(uy, vy, wy)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)


        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        #return torch.cat((uy, vy, wy), dim=1)
        return uy, vy, wy
        
        

##############################################

class Cerberus3F2U(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus3F2U, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.fme0 = FeatureMapExchange(16, 16)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()


        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.fme1 = FeatureMapExchange(32, 32)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()


        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.fme2 = FeatureMapExchange(64, 64)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()


        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.fme3 = FeatureMapExchange(128, 128)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.fme4 = FeatureMapExchange(256, 256)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.fme5 = FeatureMapExchange(512, 512)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.fme6 = FeatureMapExchange(1024, 1024)
        

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.fme7 = FeatureMapExchange(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.fme8 = FeatureMapExchange(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.fme9 = FeatureMapExchange(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.fme10 = FeatureMapExchange(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)


        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.fme11 = FeatureMapExchange(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)


        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.fme12 = FeatureMapExchange(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)


        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, input):
        
        ####### Beginning of U-ResNet #######
        #####################################
        u = input[:,:2,:,:]
        v = input[:,2:4,:,:]
        w = input[:,4:6,:,:]

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        u0, v0, w0 = self.fme0(u0, v0, w0)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        u1, v1, w1 = self.fme1(u1, v1, w1)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)


        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        u2, v2, w2 = self.fme2(u2, v2, w2)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)


        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        u3, v3, w3 = self.fme3(u3, v3, w3)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
        

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        u4, v4, w4 = self.fme4(u4, v4, w4)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
        

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        u5, v5, w5 = self.fme5(u5, v5, w5)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)
        uy, vy, wy = self.fme6(uy, vy, wy)


        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy, vy, wy = self.fme7(uy, vy, wy)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy, vy, wy = self.fme8(uy, vy, wy)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy, vy, wy = self.fme9(uy, vy, wy)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy, vy, wy = self.fme10(uy, vy, wy)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)


        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy, vy, wy = self.fme11(uy, vy, wy)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)


        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy, vy, wy = self.fme12(uy, vy, wy)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)


        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        return torch.cat((uy, vy, wy), dim=1)
        
        
        
##############################################

class Cerberus3F3(nn.Module):
    def __init__(self, n_channels, n_classes, depth=2):
        super(Cerberus3F3, self).__init__()

        self.u_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.v_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.w_re0 = ResidualX(BasicBlock, n_channels, 16, depth)
        self.fme0 = FeatureMapExchange(16, 16)
        self.u_do0 = DownX()
        self.v_do0 = DownX()
        self.w_do0 = DownX()


        self.u_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.v_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.w_re1 = ResidualX(BasicBlock, 16, 32, depth)
        self.fme1 = FeatureMapExchange(32, 32)
        self.u_do1 = DownX()
        self.v_do1 = DownX()
        self.w_do1 = DownX()


        self.u_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.v_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.w_re2 = ResidualX(BasicBlock, 32, 64, depth)
        self.fme2 = FeatureMapExchange(64, 64)
        self.u_do2 = DownX()
        self.v_do2 = DownX()
        self.w_do2 = DownX()


        self.u_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.v_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.w_re3 = ResidualX(BasicBlock, 64, 128, depth)
        self.u_do3 = DownX()
        self.v_do3 = DownX()
        self.w_do3 = DownX()


        self.u_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.v_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.w_re4 = ResidualX(BasicBlock, 128, 256, depth)
        self.u_do4 = DownX()
        self.v_do4 = DownX()
        self.w_do4 = DownX()


        self.u_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.v_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.w_re5 = ResidualX(BasicBlock, 256, 512, depth)
        self.u_do5 = DownX()
        self.v_do5 = DownX()
        self.w_do5 = DownX()


        self.u_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.v_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        self.w_re6 = ResidualX(BasicBlock, 512, 1024, depth)
        

        self.u_up0 = UpX(1024, 1024)
        self.v_up0 = UpX(1024, 1024)
        self.w_up0 = UpX(1024, 1024)
        self.u_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.v_re7 = ResidualX(BasicBlock, 1024, 512, depth)
        self.w_re7 = ResidualX(BasicBlock, 1024, 512, depth)


        self.u_up1 = UpX(512, 512)
        self.v_up1 = UpX(512, 512)
        self.w_up1 = UpX(512, 512)
        self.u_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.v_re8 = ResidualX(BasicBlock, 512, 256, depth)
        self.w_re8 = ResidualX(BasicBlock, 512, 256, depth)


        self.u_up2 = UpX(256, 256)
        self.v_up2 = UpX(256, 256)
        self.w_up2 = UpX(256, 256)
        self.u_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.v_re9 = ResidualX(BasicBlock, 256, 128, depth)
        self.w_re9 = ResidualX(BasicBlock, 256, 128, depth)


        self.u_up3 = UpX(128, 128)
        self.v_up3 = UpX(128, 128)
        self.w_up3 = UpX(128, 128)
        self.fme10 = FeatureMapExchange(128, 128)
        self.u_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.v_re10 = ResidualX(BasicBlock, 128, 64, depth)
        self.w_re10 = ResidualX(BasicBlock, 128, 64, depth)


        self.u_up4 = UpX(64, 64)
        self.v_up4 = UpX(64, 64)
        self.w_up4 = UpX(64, 64)
        self.fme11 = FeatureMapExchange(64, 64)
        self.u_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.v_re11 = ResidualX(BasicBlock, 64, 32, depth)
        self.w_re11 = ResidualX(BasicBlock, 64, 32, depth)


        self.u_up5 = UpX(32, 32)
        self.v_up5 = UpX(32, 32)
        self.w_up5 = UpX(32, 32)
        self.fme12 = FeatureMapExchange(32, 32)
        self.u_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.v_re12 = ResidualX(BasicBlock, 32, 16, depth)
        self.w_re12 = ResidualX(BasicBlock, 32, 16, depth)


        self.u_output = OutConv(16, n_classes)
        self.v_output = OutConv(16, n_classes)
        self.w_output = OutConv(16, n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) # Normalises output over the channel dimension


    def forward(self, u, v, w):
        
        ####### Beginning of U-ResNet #######
        #####################################
        #u = input[:,:2,:,:]
        #v = input[:,2:4,:,:]
        #w = input[:,4:6,:,:]

        ############ Down

        u0 = self.u_re0(u)
        v0 = self.v_re0(v)
        w0 = self.w_re0(w)
        u0, v0, w0 = self.fme0(u0, v0, w0)
        uy = self.u_do0(u0)
        vy = self.v_do0(v0)
        wy = self.w_do0(w0)
        

        u1 = self.u_re1(uy)
        v1 = self.v_re1(vy)
        w1 = self.w_re1(wy)
        u1, v1, w1 = self.fme1(u1, v1, w1)
        uy = self.u_do1(u1)
        vy = self.v_do1(v1)
        wy = self.w_do1(w1)


        u2 = self.u_re2(uy)
        v2 = self.v_re2(vy)
        w2 = self.w_re2(wy)
        u2, v2, w2 = self.fme2(u2, v2, w2)
        uy = self.u_do2(u2)
        vy = self.v_do2(v2)
        wy = self.w_do2(w2)


        u3 = self.u_re3(uy)
        v3 = self.v_re3(vy)
        w3 = self.w_re3(wy)
        uy = self.u_do3(u3)
        vy = self.v_do3(v3)
        wy = self.w_do3(w3)
        

        u4 = self.u_re4(uy)
        v4 = self.v_re4(vy)
        w4 = self.w_re4(wy)
        uy = self.u_do4(u4)
        vy = self.v_do4(v4)
        wy = self.w_do4(w4)
        

        u5 = self.u_re5(uy)
        v5 = self.v_re5(vy)
        w5 = self.w_re5(wy)
        uy = self.u_do5(u5)
        vy = self.v_do5(v5)
        wy = self.w_do5(w5)


        uy = self.u_re6(uy)
        vy = self.v_re6(vy)
        wy = self.w_re6(wy)


        ############ Up 

        uy = self.u_up0(uy, u5)
        vy = self.v_up0(vy, v5)
        wy = self.w_up0(wy, w5)
        uy = self.u_re7(uy)
        vy = self.v_re7(vy)
        wy = self.w_re7(wy)


        uy = self.u_up1(uy, u4)
        vy = self.v_up1(vy, v4)
        wy = self.w_up1(wy, w4)
        uy = self.u_re8(uy)
        vy = self.v_re8(vy)
        wy = self.w_re8(wy)


        uy = self.u_up2(uy, u3)
        vy = self.v_up2(vy, v3)
        wy = self.w_up2(wy, w3)
        uy = self.u_re9(uy)
        vy = self.v_re9(vy)
        wy = self.w_re9(wy)


        uy = self.u_up3(uy, u2)
        vy = self.v_up3(vy, v2)
        wy = self.w_up3(wy, w2)
        uy, vy, wy = self.fme10(uy, vy, wy)
        uy = self.u_re10(uy)
        vy = self.v_re10(vy)
        wy = self.w_re10(wy)


        uy = self.u_up4(uy, u1)
        vy = self.v_up4(vy, v1)
        wy = self.w_up4(wy, w1)
        uy, vy, wy = self.fme11(uy, vy, wy)
        uy = self.u_re11(uy)
        vy = self.v_re11(vy)
        wy = self.w_re11(wy)


        uy = self.u_up5(uy, u0)
        vy = self.v_up5(vy, v0)
        wy = self.w_up5(wy, w0)
        uy, vy, wy = self.fme12(uy, vy, wy)
        uy = self.u_re12(uy)
        vy = self.v_re12(vy)
        wy = self.w_re12(wy)


        ############ Post

        uy = self.u_output(uy)
        vy = self.v_output(vy)
        wy = self.w_output(wy)

        uy = self.logSoftmax(uy)
        vy = self.logSoftmax(vy)
        wy = self.logSoftmax(wy)
        #return torch.cat((uy, vy, wy), dim=1)
        return uy, vy, wy
