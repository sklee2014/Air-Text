import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.ops.deform_conv import DeformConv2d


# Calculate the output size after Conv2D
def conv2D_output_size(img_size, padding, kernel_size, stride):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


# Stack Conv2d-BatchNorm2d-Activation layers
def make_layer(i_ch, o_ch, ke, st, pd, mo=0.1):
    layer = nn.Sequential(
        nn.Conv2d(i_ch, o_ch, ke, st, pd),
        nn.BatchNorm2d(o_ch, momentum=mo),
        nn.ReLU(inplace=True)
    )
    return layer


class AirWritingModule(nn.Module):
    def __init__(self, n_class=8):
        super(AirWritingModule, self).__init__()

        self.n_class = n_class

        self.ch = [96, 96, 128, 256, 256, 256, 256, 256 + 128, 128, 5]
        self.ke = (3, 3)
        self.st = (1, 1)
        self.pd = (1, 1)

        self.conv1 = make_layer(3, self.ch[0], self.ke, self.st, self.pd)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = make_layer(self.ch[0], self.ch[1], self.ke, self.st, self.pd)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = make_layer(self.ch[1], self.ch[2], self.ke, self.st, self.pd)
        self.deconv = nn.ConvTranspose2d(self.ch[2], self.ch[2], (3, 3), (2, 2), 1, 1)
        self.conv4 = make_layer(self.ch[2], self.ch[3], self.ke, self.st, self.pd)
        self.conv5 = make_layer(self.ch[3], self.ch[4], self.ke, self.st, self.pd)
        self.conv6 = make_layer(self.ch[4], self.ch[5], self.ke, self.st, self.pd)
        self.conv7 = make_layer(self.ch[5], self.ch[6], self.ke, self.st, self.pd)
        self.conv_offset = nn.Conv2d(self.ch[7], 18, self.ke, self.st, self.pd)
        self.conv8 = DeformConv2d(self.ch[7], self.ch[8], 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.ch[8])
        self.relu = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(self.ch[8], self.ch[9], self.ke, self.st, self.pd)
        self.classifier_conv = nn.Sequential(
            nn.Conv2d(384, 128, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (0, 1)),
            nn.ReLU(inplace=True)
        )
        self.classifier_lin = nn.Sequential(
            nn.Linear(128 * 7 * 10, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.n_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        xd = self.deconv(x)
        x = self.conv4(xd)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.cat((x, xd), dim=1)
        offsets = self.conv_offset(x)
        xh = self.conv8(x, offsets)
        xh = self.bn(xh)
        xh = self.relu(xh)
        xh = self.conv9(xh)
        xc = self.classifier_conv(x)
        xc = torch.flatten(xc, 1)
        xc = self.classifier_lin(xc)

        return xh, xc


if __name__ == "__main__":
    input = torch.randn(8, 3, 120, 160)
    y = AirWritingModule()
    z = y(input)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = y.to(device)
    summary(y, (3, 120, 160))