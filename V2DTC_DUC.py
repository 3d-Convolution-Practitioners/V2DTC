import torch.nn as nn
import torch

class PixelShuffle2D(nn.Module):


    def __init__(self, scale_factor, is_reverse=False):
        """
        :param scale_factor(int,list,tuple): Scale up/down factor, if the input scale_factor is int,
         x,y axes of a data will scale up/down with the same scale factor,
         else x,y axes of a data will scale with different scale factor
        :param is_reverse(bool): True for TC2D, False for DUC.
        """
        if isinstance(scale_factor, int):
            self.scale_factor_x = self.scale_factor_y = scale_factor
        elif isinstance(scale_factor, tuple) or isinstance(scale_factor, list):
            self.scale_factor_x = scale_factor[0]
            self.scale_factor_y = scale_factor[1]
        else:
            print("scale factor should be int or tuple or list")
            raise ValueError
        super(PixelShuffle2D, self).__init__()
        self.is_reverse = is_reverse

    def forward(self, inputs):
        batch_size, channels, in_height, in_width = inputs.size()
        if self.is_reverse:  # for Variable 2D Tiled Convolution
            out_channels = channels * self.scale_factor_x * self.scale_factor_y
            out_height = in_height // self.scale_factor_x
            out_width  = in_width  // self.scale_factor_y
            input_view = inputs.contiguous().view(
                batch_size, channels,
                out_height, self.scale_factor_x,
                out_width , self.scale_factor_y)
            shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
            return shuffle_out.view(batch_size, out_channels, out_height, out_width)
        else:  # for Dense Upsampling Convolution
            channels //= (
                        self.scale_factor_x * self.scale_factor_y)
            # out channels, it should equal to class number for segmentation task

            out_height = in_height * self.scale_factor_x
            out_width  = in_width  * self.scale_factor_y

            input_view = inputs.contiguous().view(
                batch_size, channels,
                self.scale_factor_x, self.scale_factor_y,
                in_height,           in_width)

            shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
            return shuffle_out.view(batch_size, channels, out_height, out_width)


class HDC(nn.Module):
    def __init__(self, downscale_factor):
        """
        reference paper: Zeng, G., & Zheng, G. (2019). Holistic decomposition convolution for effective semantic
         segmentation of medical volume images. Medical image analysis, 57, 149-164.

        2D Tiled Convolution module, the input data dimensions should be 4D tensor like (batch, channel, x, y, z),
        :param downscale_factor(int, tuple, list): Scale down factor, if the input scale_factor is int,
         x,yaxes of a data will scale down with the same scale factor,
         else x,y axes of a data will scale with different scale factor
        """

        super(HDC, self).__init__()
        self.ps = PixelShuffle2D(downscale_factor, is_reverse=True)

    def forward(self, x):
        x = self.ps(x)
        return x


class DUC(nn.Module):

    def __init__(self, upscale_factor, class_num, in_channels):
        """
        reference paper: Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016).
         Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network.
          In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1874-1883).

        2D DUC module, the input data dimensions should be 4D tensor like(batch, channel, x, y),
        workflow: conv->batchnorm->relu->pixelshuffle

        :param upscale_factor(int, tuple, list): Scale up factor, if the input scale_factor is int,
         x,y axes of a data will scale up with the same scale factor,
         else x,y axes of a data will scale with different scale factor
        :param class_num(int): the number of total classes (background and instance)
        :param in_channels(int): the number of input channel
        """
        super(DUC, self).__init__()
        if isinstance(upscale_factor, int):
            scale_factor_x = scale_factor_y = upscale_factor
        elif isinstance(upscale_factor, tuple) or isinstance(upscale_factor, list):
            scale_factor_x = upscale_factor[0]
            scale_factor_y = upscale_factor[1]
        else:
            print("scale factor should be int or tuple")
            raise ValueError
        self.conv = nn.Conv2d(in_channels, class_num * scale_factor_x * scale_factor_y, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(class_num * scale_factor_x * scale_factor_y)
        self.relu = nn.ReLU(inplace=True)
        self.ps = PixelShuffle2D(upscale_factor, is_reverse=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ps(x)
        return x