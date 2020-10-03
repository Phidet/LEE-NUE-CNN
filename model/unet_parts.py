""" Parts of the U-Net model """

import torch
import torch.nn as nn


#####################################
# U-Net parts
#####################################
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DownX(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxPool(x)
        return x

class UpX(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()
        # Instantiate the stack of residual units
        self.convTrans42 = nn.ConvTranspose2d(inplanes, outplanes//2, kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.convTrans42(x)
        x = torch.cat((x, skip), dim=1)
        self.relu(x)
        return x


#####################################
# ResNet parts adfapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#####################################

class ResidualX(nn.Module):
    def __init__(self, block, inplanes, outplanes, layers, stride=1, zero_init_residual=False, groups=1, width_per_group=64):
        super(ResidualX, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = inplanes
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.residual = self._make_layer(block, outplanes, layers, stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, outplanes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, outplanes * block.expansion, stride),
                norm_layer(outplanes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = outplanes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, outplanes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.residual(x)
        return x




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



#####################################
# Cerberus parts
#####################################


class FeatureMapExchange(nn.Module):
	# The FeatureMapExchange takes feature stacks from three U-Nets/U-ResNets with width W, height H anc C channels.
    # It seperately applies a 2D convolution to each while reducing the other two along a given image axis using maxpool.
    # This reduction produces two 1xHxC tensors to which a 1D convolution is applied. They are then repeated along 
    # the reduced axis to get back to a WxHxC tensor. The two tensors are then added to the one with the 2D convolution. 
    # Overall this happens three times, once for each input feature map.
    # 
    # The axis along which the images (feature stacks) are reduced has to be shared along all images.
    # Specifically, this means that the images can be taken from different perspectives as long as all three perspectives 
    # only vary by a shared rotation axis (that is parallel to either the rows or the columns in the images). 
    # In other words, the cameras need to rotate around a cylyindrical volume. The z-axis of the cylinder is then the unaltered dimension.       

    def __init__(self, in_planes, out_planes):
        super(FeatureMapExchange, self).__init__()
        self.u_conv1D = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.v_conv1D = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.w_conv1D = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.u_conv3x3 = conv3x3(in_planes, out_planes)
        self.v_conv3x3 = conv3x3(in_planes, out_planes)
        self.w_conv3x3 = conv3x3(in_planes, out_planes)

        self.bn_0 = nn.BatchNorm2d(out_planes)
        self.bn_1 = nn.BatchNorm2d(out_planes)
        self.bn_2 = nn.BatchNorm2d(out_planes)
        
        self.bn1D_0 = nn.BatchNorm1d(out_planes)
        self.bn1D_1 = nn.BatchNorm1d(out_planes)
        self.bn1D_2 = nn.BatchNorm1d(out_planes)
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, u, v, w):
        # size of the maps along the non-shared axis (MicroBooNE Z) 
        numZ = u.size()[2]

        u_feature_vector, _ = torch.max(u, keepdim=False, dim=3)
        v_feature_vector, _ = torch.max(v, keepdim=False, dim=3)
        w_feature_vector, _ = torch.max(w, keepdim=False, dim=3)

        u_feature_vector = self.u_conv1D(u_feature_vector)
        v_feature_vector = self.v_conv1D(v_feature_vector)
        w_feature_vector = self.w_conv1D(w_feature_vector)
        
        u_feature_vector = self.bn1D_0(u_feature_vector)
        v_feature_vector = self.bn1D_1(v_feature_vector)
        w_feature_vector = self.bn1D_2(w_feature_vector)

        
        self.u_conv3x3(u)
        self.v_conv3x3(v)
        self.w_conv3x3(w)
        
        u = self.bn_0(u)
        v = self.bn_1(v)
        w = self.bn_2(w)
        
        u = self.relu(u)
        v = self.relu(v)
        w = self.relu(w)
        
        u_feature_vector = self.relu(u_feature_vector)
        v_feature_vector = self.relu(v_feature_vector)
        w_feature_vector = self.relu(w_feature_vector)

        u_exchange_map = u_feature_vector[:,:,:,None].expand(-1,-1,-1,numZ)
        v_exchange_map = v_feature_vector[:,:,:,None].expand(-1,-1,-1,numZ)
        w_exchange_map = w_feature_vector[:,:,:,None].expand(-1,-1,-1,numZ)

        u += v_exchange_map + w_exchange_map
        v += w_exchange_map + u_exchange_map
        w += u_exchange_map + v_exchange_map

        return u, v, w
