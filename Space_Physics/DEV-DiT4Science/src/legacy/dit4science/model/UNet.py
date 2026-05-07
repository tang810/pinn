# Author: Chunyang Wang; Github Username: Chunyang-w
# E-mail: chunyang.wang22@imperial.ac.uk
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DownBlock(nn.Module):
    """
    The Down-sample convolution Block.
    The output size of the convolution block can be calculated using:
    ``` Output_Size= {[ Input_Size − Kernel_Size+ 2*Padding ] / Stride} + 1 ```
    Note that in downsample path, we typically want the size shrink by
    the factor of 2. So stride is usually set to 2.

    Arguments are self-explanatory. bn is the batch norm switch.
    """
    def __init__(self,
                 in_c, out_c,
                 kn_size=4, padding=1, stride=2,
                 drop=0., bn=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c,
                kernel_size=kn_size, padding=padding, stride=stride,
            )
        )
        if bn is True:
            self.block.add_module(
                name="batch norm", module=nn.BatchNorm2d(out_c))
        self.block.add_module(
            name="drop out", module=nn.Dropout2d(drop, inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    """
    The Up-sample convolution Block.
    The output size of the convolution block can be calculated using:
    ``` Output_Size= {[ Input_Size − Kernel_Size+ 2*Padding ] / Stride} + 1 ```
    Note that in downsample path, we typically want the size inflated by
    the factor of 2. So a Upsample block will inflate the size by a
    scale factor of 2, Then the input will go through a conv layer which
    keeps the size unchanged - So stride is set to 1 here by default.

    Arguments are self-explanatory. bn is the batch norm switch.
    """
    def __init__(self,
                 in_c, out_c,
                 kn_size=3, padding=1, stride=1,
                 drop=0., bn=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c,
                kernel_size=kn_size, padding=padding, stride=stride,
            )
        )
        if bn is True:
            self.block.add_module(name="batch norm", module=nn.BatchNorm2d(out_c))  # noqa
        self.block.add_module(
            name="drop out", module=nn.Dropout(drop, inplace=True))

    def forward(self, x):
        x = self.block(x)
        return x


class UNet(nn.Module):
    """
    A Unet that has 4 encode layers, each layer shrink the input by
    a scale of 2. The decoder layers will restore the final encoded
    tensor to the original shape. (This is the default behaviour, the
    architecture can be instantiated by altering parameters when
    initialising the object)

    The encoder is a convolutional layer along with LeakyReLU, batch-norm
    and dropout.

    The decoder is a convolutional layer with Upscale (bilinear) layer and
    batch-norm layer.

    The decoded tensor is concatenated with residual connection with its
    downscaleing counterpart before going to next decoder layer.

    e.g, if the input is of shape (16 * 16 * 4):
    encoding procedure:
        (16*16*4) -> (8*8*8)
        (8*8*8)   -> (4*4*16)
        (4*4*16)  -> (2*2*32)
        (2*2*32)  -> (1*1*64)
    decoding procddure:
        (1*1*64) -> (2*2*32)
        (2*2*64) -> (4*4*16)
        (4*4*32) -> (8*8*8)
        (8*8*16) -> (16*16*4)
    """

    def __init__(self,
                 channel_list=[4, 32, 64, 128, 512],
                 kn_size_down=4, padding_down=1, stride_down=2,
                 kn_size_up=3, padding_up=1, stride_up=1,
                 drop=0., bn=True,
                 ):
        """Will only provide docs for channel_list. Other params should be
        pretty self-explanatory.

        Args:
            channel_list: a list containing number of channels in each of
            the encoder conv layer, including the number of channels of
            the original input. If the input is of shape (128*128*4), and
            the final encoded tensor is of shape (8*8*128), channel_list
            should be something like: [4, 16, 32, 64, 128]
        """
        super().__init__()
        self.num_layers = len(channel_list) - 1
        encode_layers = [
            [*t] for t in zip(
                channel_list[0:-1], channel_list[1:]
            )
        ]
        decode_layers = [
            [t[1]*2, t[0]] for t in encode_layers[::-1]
        ]
        decode_layers[0][0] = encode_layers[-1][-1]
        self.encoder_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoder_list.append(
                DownBlock(
                    in_c=encode_layers[i][0], out_c=encode_layers[i][1],
                    kn_size=kn_size_down, padding=padding_down, stride=stride_down,  # noqa
                    drop=drop, bn=bn,
                )
            )
            self.decoder_list.append(
                UpBlock(
                    in_c=decode_layers[i][0], out_c=decode_layers[i][1],
                    kn_size=kn_size_up, padding=padding_up, stride=stride_up,  # noqa
                    drop=drop, bn=bn,
                )
            )
        # print(self.encoder_list)
        # print(self.decoder_list)

    def forward(self, x, return_latents=False):
        encoded = []
        decoded = []
        for i in range(self.num_layers):
            x = self.encoder_list[i](x)
            encoded.append(x)
        for i in range(self.num_layers):
            if i == 0:
                x = self.decoder_list[i](x)
            else:
                x = torch.cat([x, encoded[-(i+1)]], dim=1)
                x = self.decoder_list[i](x)
            decoded.append(x)
        return x if not return_latents else (x, encoded, decoded)
