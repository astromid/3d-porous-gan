from abc import abstractmethod
from math import log2
from typing import NoReturn

import torch
import torch.nn as nn


class GANModule(nn.Module):
    """
    A base class for both generator and the discriminator.
    Provides a common weight initialization scheme.
    """
    def weights_init(self) -> NoReturn:
        """
        Initialization scheme from DCGAN paper.
        """
        for m in self.modules():
            layer_class = m.__class__.__name__

            if 'Conv' in layer_class:
                m.weight.data.normal_(0.0, 0.02)

            elif 'BatchNorm' in layer_class:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Generator(GANModule):
    """ 
    DCGAN-like generator network with 3D convolutions.
    """
    def __init__(self, img_size: int, z_dim: int, num_channels: int, num_filters: int, num_extra_layers: int = 0):
        """
        :param int img_size: size of the generated image.
        :param int z_dim: dimension of latent space (noise vector).
        :param int num_channels: number of channels in output image.
        :param int num_filters: number of filters in the second-to-last deconvolutional layer.
        :param int num_extra_layers: number of extra convolution blocks.
        """
        super().__init__()

        if not (img_size % 16 == 0):
            raise ValueError(f"Image size has to be a multiple of 16, got {img_size}")

        # calculate corresponding number of deconvolution blocks and filters in 1st deconvolution block
        # img_size == 16 * factor; img_size / 2 == 4 * 2**n => n = 1 + log2(factor)
        factor = img_size / 16
        n = int(1 + log2(factor))
        current_num_filters = int(num_filters * 2**n)

        # build 1st deconvolution block
        self.net = nn.Sequential(
            # input is z (noise), going into a deconvolution
            nn.ConvTranspose3d(
                in_channels=z_dim,
                out_channels=current_num_filters,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm3d(num_features=current_num_filters),
            nn.ReLU(inplace=True)
        )
        # shape: (current_num_filters x 4 x 4)

        # for proper layers naming
        layers_counter = 3

        # build deconvolution blocks
        for i in range(n):
            current_num_filters //= 2
            self.net.add_module(
                name=str(layers_counter),
                module=nn.ConvTranspose3d(
                    in_channels=current_num_filters*2,
                    out_channels=current_num_filters,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.net.add_module(
                name=str(layers_counter+1),
                module=nn.BatchNorm3d(num_features=current_num_filters)
            )
            self.net.add_module(
                name=str(layers_counter+2),
                module=nn.ReLU(inplace=True)
            )
            layers_counter += 3

        # build extra deconvolutional blocks
        for i in range(num_extra_layers):
            self.net.add_module(
                name=str(layers_counter),
                module=nn.Conv3d(
                    in_channels=current_num_filters,
                    out_channels=current_num_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            self.net.add_module(
                name=str(layers_counter+1),
                module=nn.BatchNorm3d(num_features=current_num_filters)
            )
            self.net.add_module(
                name=str(layers_counter+2),
                module=nn.ReLU(inplace=True)
            )
            layers_counter += 3

        # output deconvolution block
        self.net.add_module(
            name=str(layers_counter),
            module=nn.ConvTranspose3d(
                in_channels=current_num_filters,
                out_channels=num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        self.net.add_module(
            name=str(layers_counter+1),
            module=nn.Tanh()
        )
        self.weights_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(GANModule):
    """
    DCGAN-like discriminator network with 3D convolutions.
    """
    def __init__(self, img_size: int, num_channels: int, num_filters: int, num_extra_layers: int = 0):
        """
        :param int img_size: size of the generated image.
        :param int num_channels: number of channels in output image.
        :param int num_filters: number of filters in the first convolutional layer.
        :param int num_extra_layers: number of extra convolution blocks.
        """
        super().__init__()

        if not (img_size % 16 == 0):
            raise ValueError(f"Image size has to be a multiple of 16, got {img_size}")

        # build first convolutional block
        self.net = nn.Sequential(
            # input shape: (num_channels x img_size x img_size)
            nn.Conv3d(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # for proper layers naming
        layers_counter = 2

        # build extra convolutional blocks
        for i in range(num_extra_layers):
            self.net.add_module(
                name=str(layers_counter),
                module=nn.Conv3d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            self.net.add_module(
                name=str(layers_counter+1),
                module=nn.BatchNorm3d(num_filters)
            )
            self.net.add_module(
                name=str(layers_counter+2),
                module=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            layers_counter += 3

        # calculate corresponding number of convolution blocks
        # img_size == 16 * factor; img_size / 2 == 4 * 2**n => n = 1 + log2(factor)
        factor = img_size / 16
        n = int(1 + log2(factor))

        # build convolutional blocks
        for i in range(n):
            self.net.add_module(
                name=str(layers_counter),
                module=nn.Conv3d(
                    in_channels=num_filters,
                    out_channels=num_filters*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.net.add_module(
                name=str(layers_counter+1),
                module=nn.BatchNorm3d(num_features=num_filters*2)
            )
            self.add_module(
                name=str(layers_counter+2),
                module=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            layers_counter += 3
            num_filters *= 2

        # shape: (K x 4 x 4 x 4)
        self.net.add_module(
            name=str(layers_counter),
            module=nn.Conv3d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )
        self.net.add_module(
            name=str(layers_counter+1),
            module=nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output.view(-1, 1).squeeze(1)
