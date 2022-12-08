import torch
import torchvision
import torch.nn.functional as F
from torch import nn

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



# High Frequency Extractor,(HFE)
class HFE(nn.Module):
    def __init__(self, in_channels, gamma, scale_facotr=2):
        super(HFE, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = gamma
        self.scale_factor = scale_facotr

    def forward(self, shallow, deep):
        deep = self.conv1x1(deep)
        f_h = torch.subtract(shallow, deep)
        out = torch.add(shallow, f_h, alpha=self.gamma)
        return out




# Multiscale Convolution layers

class MSC(nn.Module):
    def __init__(self, in_channels):
        super(MSC, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.m1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.m2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.m3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.m4 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.m5 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        self.conv_out1x1 = nn.Conv2d(6 * in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        y1 = self.m1(x)
        y2 = self.m2(x + y1)
        y3 = self.m3(x + y2)
        y4 = self.m4(x + y3)
        y5 = self.m5(x + y4)
        y = torch.cat([y1, y2, y3, y4, y5, x], dim=1)
        return self.conv_out1x1(y)



# Dilated Atrous Convolution

class DAC(nn.Module):

    def __init__(self, channels):
        super(DAC, self).__init__()
        self.conv11 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)

        self.conv21 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv22 = nn.Conv2d(channels, channels, kernel_size=1, dilation=1, padding=0)

        self.conv31 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.conv32 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv33 = nn.Conv2d(channels, channels, kernel_size=1, dilation=1, padding=0)

        self.conv41 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        self.conv42 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        self.conv43 = nn.Conv2d(channels, channels, kernel_size=3, dilation=5, padding=5)
        self.conv44 = nn.Conv2d(channels, channels, kernel_size=1, dilation=1, padding=0)

    def forward(self, x):
        c1 = F.relu(self.conv11(x))

        c2 = self.conv21(x)
        c2 = F.relu(self.conv22(c2))

        c3 = self.conv31(x)
        c3 = self.conv32(c3)
        c3 = F.relu(self.conv33(c3))

        c4 = self.conv41(x)
        c4 = self.conv42(c4)
        c4 = self.conv43(c4)
        c4 = F.relu(self.conv44(c4))

        c = x + c1 + c2 + c3 + c4

        return c


# Residual Multi Kernel Pooling

class RMP(nn.Module):

    def __init__(self, channels):
        super(RMP, self).__init__()

        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(channels, 1, kernel_size=1)

        self.max2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size=1)

        self.max3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv3 = nn.Conv2d(channels, 1, kernel_size=1)

        self.max4 = nn.MaxPool2d(kernel_size=6)
        self.conv4 = nn.Conv2d(channels, 1, kernel_size=1)
        # modified
        self.out = nn.Conv2d(channels+4, channels, kernel_size=1)
    def forward(self, x):
        # print(x.size())
        m1 = self.max1(x)
        # print(m1.size())
        m1 = F.interpolate(self.conv1(m1), size=x.size()[2:], mode='bilinear', align_corners=True)
        # print(m1.size())
        m2 = self.max2(x)
        m2 = F.interpolate(self.conv2(m2), size=x.size()[2:], mode='bilinear', align_corners=True)

        m3 = self.max3(x)
        m3 = F.interpolate(self.conv3(m3), size=x.size()[2:], mode='bilinear', align_corners=True)

        m4 = self.max4(x)
        m4 = F.interpolate(self.conv4(m4), size=x.size()[2:], mode='bilinear', align_corners=True)## confused whether true or false

        m = torch.cat([m1, m2, m3, m4, x], axis=1)
        # print(m.size())
        return self.out(m)

# Reference code: https://github.com/khanhha/crack_segmentation/blob/e924b6a3632134848b993c68e7295b1aae92ce28/unet/unet_transfer.py#L39
class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, up_sampling_method="interpolate"):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if up_sampling_method == "deconv":
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )

        elif up_sampling_method == "pixel_shuffle":
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )

        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, up_sampling_method="pixel_shuffle", dac=False, receptive_enlarge=False, re_method:str = None):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        # print(torchvision.models.vgg16(pretrained=pretrained))

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)
        self.dac = dac
        self.DAC = DAC(512)
        self.receptive_enlarge = receptive_enlarge
        self.re_method = re_method
        self.RMP = RMP(512)
        self.MSC = MSC(512)
        self.center = DecoderBlockV2(512, num_filters * 8 * 4, num_filters * 8, up_sampling_method)

        # self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 4, num_filters * 8, up_sampling_method)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 4, num_filters * 4, up_sampling_method)
        self.dec2 = DecoderBlockV2(128 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, up_sampling_method)
        self.dec1 = ConvRelu(64 + num_filters * 2, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        if self.dac:
            conv5 = self.DAC(conv5)
        if self.receptive_enlarge:
            if self.re_method == 'msc':
                conv5 = self.MSC(conv5)
            elif self.re_method == 'rmp':
                conv5 = self.RMP(conv5)

        center = self.center(conv5)

        dec4 = self.dec4(torch.cat([center, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.sigmoid(self.final(dec1))

        return x_out


if (__name__ == "__main__"):
    # input_data = torch.randn(2, 3, 400, 400)
    # model = MSC(in_channels=3)
    # output = model(input_data)
    # HFE = HFE(in_channels=3, gamma=0.1)
    # output = HFE(input_data, output)
    input_data = torch.randn(2, 3, 400, 400)
    model = UNet16(pretrained=True, dac=True, re_method='rmp', receptive_enlarge=True)
    output = model(input_data)
    print(output.shape)
