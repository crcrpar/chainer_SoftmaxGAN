# !/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import chainer
import chainer.functions as F


class SoftmaxGANUpdater(chainer.updater.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(SoftmaxGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        opt_gen = self.get_optimizer('gen')
        opt_dis = self.get_optimier('dis')
        gen = self.gen
        dis = self.dis
        xp = gen.xp

        batch = self.get_iterator('main').next()
        batch_size = len(batch)

        x_real = chainer.Variable(self.converter(batch, self.device)) / 255.0
        y_real = dis(x_real, test=False)

        z = chainer.Variable(xp.asarray(gen.make_hidden(batch_size)))
        x_fake = gen(z)
        y_fake = dis(x_fake, test=False)

        loss_D = F.sum(y_real) / (batch_size * batch_size) + F.sum(F.log(y_real + y_fake))
        loss_G = F.sum(y_real + y_fake) / (2 * batch_size * batch_size) + F.sum(F.log(y_real + y_fake))

        chainer.report({'loss': loss_D}, dis)
        chainer.report({'loss': loss_G}, gen)

        dis.cleargrads()
        loss_D.backward()
        opt_dis.update()

        gen.cleargrads()
        loss_G.backward()
        opt_gen.update()
