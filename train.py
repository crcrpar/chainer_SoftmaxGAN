#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import yaml

import chainer
from chainer import training
from chainer.training import extensions

from updater import SoftmaxGANUpdater
from visualizer import out_generated_image


def main():
    with open('setting.yml', 'r') as f:
        conf = yaml.load(f)
    if conf['bn']:
        import models.with_bn as nets
    else:
        import models.no_bn as nets
    gen = nets.Generator
    dis = nets.Discriminator
    if conf['gpu'] >= 0:
        chainer.cuda.get_device(conf['gpu']).use()
        gen.to_gpu()
        dis.to_gpu()
    opt_gen = chainer.optimizers.Adam()
    opt_dis = chainer.optimizers.Adam()
    opt_gen.setup(gen)
    opt_dis.setup(dis)

    dataset = chainer.datasets.ImageDataset(conf['data'])
    if conf['parallel']:
        iterator = chainer.iterators.MultiprocessIterator(
            dataset, conf['batch_size'])
    else:
        iterator = chainer.iterators.SerialIterator(dataset, conf['batch_size'])
    updater = SoftmaxGANUpdater(
        models=(gen, dis),
        iterator=iterator,
        optimizer={
            'gen': opt_gen, 'dis': opt_dis},
        device=conf['gpu'])
    trainer = training.Trainer(
        updater, (conf['eppoch'], 'epoch'), out=conf['out']) 
    snapshot_interval = (conf['snapshot_interval'], 'epoch')
    display_interval = (conf['display_interval'], 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, dis,
            10, 10, conf['seed'], conf['out']),
        trigger=snapshot_interval)
    if conf['resume']:
        chainer.serializers.load_npz(conf['resume'], trainer)

    trainer.run()


if __name__ == '__main__':
    main()
