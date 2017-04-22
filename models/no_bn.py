# !/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import numpy

import chainer
import chainer.functions as F
import chainer.links as L


def add_noise(h, test, sigma=0.2):
    xp = chainer.cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)


class Generator(chainer.Chain):

    def __init__(self, n_hidden, bottom_width=4, ch=512):
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        W = chainer.initializers.HeNormal()
        kwargs = {
            'ksize': 4,
            'stride': 2,
            'pad': 1,
            'initialW': chainer.initializers.HeNormal()}
        self._layers = {
            'l0': L.Linear(self.n_hidden, ch * bottom_width * bottom_width, initialW=W),
            'dc1': L.Deconvolution2D(ch, ch // 2, **kwargs),
            'dc2': L.Deconvolution2D(ch // 2, ch // 4, **kwargs),
            'dc3': L.Deconvolution2D(ch // 4, ch // 8, **kwargs),
            'dc4': L.Deconvolution2D(ch // 8, 3, ksize=3, stride=3, pad=1, initialW=W)}
        super(Generator, self).__init__(**self._layers)

    def make_hidden(self, batchsize):
        h = numpy.random.uniform(
            low=-1.0, high=1.0, size=(batchsize, self.n_hidden, 1, 1))\
            .astype(numpy.float32)
        return h

    def __call__(self, z):
        h = F.reshape(F.relu(self.l0(z)), (z.data.shape[
                      0], self.ch, self.bottom_width, self.bottom_width))
        h = F.relu(self.dc1(h))
        h = F.relu(self.dc2(h))
        h = F.relu(self.dc3(h))
        x = F.sigmoid(self.dc4(h))
        return x


class Discriminator(chainer.Chain):

    def __init__(self, bottom_width=4, ch=512):
        W = chainer.initializers.HeNormal()
        kwargs_0 = {
            'ksize': 3,
            'stride': 1,
            'pad': 1,
            'initialW': W}
        kwargs_1 = {
            'ksize': 4,
            'stride': 2,
            'pad': 1,
            'initialW': W}
        self._layers = {
            'c0_0': L.Convolution2D(3, ch // 8, **kwargs_0),
            'c0_1': L.Convolution2D(ch // 8, ch // 4, **kwargs_1),
            'c1_0': L.Convolution2D(ch // 4, ch // 4, **kwargs_0),
            'c1_1': L.Convolution2D(ch // 4, ch // 2, **kwargs_1),
            'c2_0': L.Convolution2D(ch // 2, ch // 2, **kwargs_0),
            'c2_1': L.Convolution2D(ch // 2, ch // 1, **kwargs_1),
            'c3_0': L.Convolution2D(ch // 1, ch // 1, **kwargs_0),
            'l4': L.Linear(ch * bottom_width * bottom_width, 1, initialW=W)}
        super(Discriminator, self).__init__(**self._layers)

    def __call__(self, x, test=False):
        h = add_noise(x, test=test)
        h = F.leaky_relu(add_noise(self.c0_0(h), test=test))
        h = F.leaky_relu(add_noise(self.c0_1(h), test=test))
        h = F.leaky_relu(add_noise(self.c1_0(h), test=test))
        h = F.leaky_relu(add_noise(self.c1_1(h), test=test))
        h = F.leaky_relu(add_noise(self.c2_0(h), test=test))
        h = F.leaky_relu(add_noise(self.c2_1(h), test=test))
        h = F.leaky_relu(add_noise(self.c3_0(h), test=test))
        return self.l4(h)
