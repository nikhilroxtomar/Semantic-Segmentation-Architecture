
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, act=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        if act == True:
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class multires_block(nn.Module):
    def __init__(self, in_c, out_c, alpha=1.67):
        super().__init__()

        W = out_c * alpha
        self.c1 = conv_block(in_c, int(W*0.167))
        self.c2 = conv_block(int(W*0.167), int(W*0.333))
        self.c3 = conv_block(int(W*0.333), int(W*0.5))

        nf = int(W*0.167) + int(W*0.333) + int(W*0.5)
        self.b1 = nn.BatchNorm2d(nf)
        self.c4 = conv_block(in_c, nf)
        self.relu = nn.ReLU(inplace=True)
        self.b2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        x0 = x
        x1 = self.c1(x0)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        xc = torch.cat([x1, x2, x3], dim=1)
        xc = self.b1(xc)

        sc = self.c4(x0)
        x = self.relu(xc + sc)
        x = self.b2(x)
        return x

class res_path_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = conv_block(in_c, out_c, act=False)
        self.s1 = conv_block(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.c1(x)
        s1 = self.s1(x)
        x = self.relu(x1 + s1)
        x = self.bn(x)
        return x

class res_path(nn.Module):
    def __init__(self, in_c, out_c, length):
        super().__init__()

        layers = []
        for i in range(length):
            layers.append(res_path_block(in_c, out_c))
            in_c = out_c

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

def cal_nf(ch, alpha=1.67):
    W = ch * alpha
    return int(W*0.167) + int(W*0.333) + int(W*0.5)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, length):
        super().__init__()

        self.c1 = multires_block(in_c, out_c)
        nf = cal_nf(out_c)
        self.s1 = res_path(nf, out_c, length)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        s = self.s1(x)
        p = self.pool(x)
        return s, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.ConvTranspose2d(in_c[0], out_c, kernel_size=2, stride=2, padding=0)
        self.c2 = multires_block(out_c+in_c[1], out_c)

    def forward(self, x, s):
        x = self.c1(x)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)
        return x

class build_multiresunet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 32, 4)
        self.e2 = encoder_block(cal_nf(32), 64, 3)
        self.e3 = encoder_block(cal_nf(64), 128, 2)
        self.e4 = encoder_block(cal_nf(128), 256, 1)

        """ Bridge """
        self.b1 = multires_block(cal_nf(256), 512)

        """ Decoder """
        self.d1 = decoder_block([cal_nf(512), 256], 256)
        self.d2 = decoder_block([cal_nf(256), 128], 128)
        self.d3 = decoder_block([cal_nf(128), 64], 64)
        self.d4 = decoder_block([cal_nf(64), 32], 32)

        """ Output """
        self.output = nn.Conv2d(cal_nf(32), 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b1 = self.b1(p4)

        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        output = self.output(d4)
        return output

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = build_multiresunet()
    output = model(x)
    print(output.shape)
