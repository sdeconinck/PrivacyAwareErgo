import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    """
        AutoEncoder class, based on the MobileNet architecture
    """

    def __init__(self, internal_expansion=6, first_n_filters=32, transforms=None, upscale_factor=(2,2,2), random_map=False):
        """Constructor

        Args:
            transforms ([Transformation], optional): [Transformation that should be applied to input images]. Defaults to None.
        """
        super(AutoEncoder, self).__init__()
        self.random_map = random_map
        self.transforms = transforms
        self.conv1 = nn.Conv2d(3, first_n_filters, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.bn1 = nn.InstanceNorm2d(first_n_filters,  eps=1e-4)

        self.encoder = nn.Sequential(
            MobilenetV2Block(first_n_filters, first_n_filters, stride=upscale_factor[0], transposed=False, expansion=internal_expansion),
            MobilenetV2Block(first_n_filters, int(1.5 * first_n_filters), stride=upscale_factor[1], transposed=False, expansion=internal_expansion),
            MobilenetV2Block(int(1.5 * first_n_filters), first_n_filters *2 , stride=upscale_factor[2], transposed=False, expansion=internal_expansion),
        )

        n_output_layers = 4 if random_map else 3
        self.decoder = nn.Sequential(
            MobilenetV2Block(first_n_filters*2,int(first_n_filters*1.5), 1, True, expansion=internal_expansion, upscale_factor=upscale_factor[2]),
            MobilenetV2Block(int(first_n_filters*1.5), first_n_filters, 1, True, expansion=internal_expansion, upscale_factor=upscale_factor[1]),
            MobilenetV2Block( first_n_filters,n_output_layers, 1, True, expansion=internal_expansion, upscale_factor=upscale_factor[0]),
        )

        # sigmoid to revert values to the [0,1] scale for processing with the object detection model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.transforms != None:
            x = self.transforms(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        h = self.encoder(x)
        out = self.decoder(h)
        out = self.sigmoid(out)

        if self.random_map:
            # split up output into the random map and the image
            img = out[:,0:3,:,:]
            random_map = out[:,3:,:,:]

            random_img = torch.rand_like(img)
            random_img = img + random_img * random_map
            return self.sigmoid(random_img), random_map

        return out



# for the mobilenet encoder
class MobilenetV2Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, stride, transposed=False, expansion=6, upscale_factor=2):
        super(MobilenetV2Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn1 = nn.InstanceNorm2d(planes,  eps=1e-4)
        if not transposed:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=stride, padding=1, groups=planes, bias=True)
        else:
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1, output_padding=0, groups=planes,
                                   bias=True),
                nn.Upsample(scale_factor=upscale_factor),
                #nn.LeakyReLU()
            )
        self.bn2 = nn.InstanceNorm2d(planes,  eps=1e-4)
        self.conv3 = nn.Conv2d(
            planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.bn3 = nn.InstanceNorm2d(out_planes,  eps=1e-4)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out = self.bn3(self.conv3(out))
        return out


# code from https://github.com/milesial/Pytorch-UNet for the UNet network
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
