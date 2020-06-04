import numpy as np
import tensorflow as tf
import os
import random

import data
import utils
import network
import params


if __name__ == '__main__':
    # 读取tfrecord数据集
    dataset = data.read_sorted_tfrecord()
    
    # 如果在训练autoencoder
    if params.cur_step == 'ae':
        Autoencoder = network.Autoencoder()

        for epoch in range(params.epochs):
            iters = 0

            for x, y in dataset:
                iters += 1
                print('epoch ', epoch, ' iteration ', iters)

                Autoencoder.step(y)

                if (iters % 200 == 0):
                    Autoencoder.validate(y, '{}_{}'.format(epoch, iters))
        
            Autoencoder.save_model()

    # 如果在匹配image encoder与voxel encoder
    elif params.cur_step == 'com':
        Combiner = network.Combiner(training =True)
        
        for epoch in range(params.epochs):
            iters = 0

            for x, y in dataset:
                iters += 1

                y_true = Combiner.voxel_encoder(y, training = False)

                Combiner.init_step(x[:, 0, :, :, :], y_true)
                used = [0]

                for t in range(params.time_steps - 1):
                    enc_loss = []
                    for i in range(24):
                        loss = 999999
                        if i not in used:
                            loss = Combiner.encoder_step(x[:, i, :, :, :], y_true)
                        enc_loss.append(loss)
                    act_true = np.zeros(24)
                    act_true[np.argmin(enc_loss)] = 1
                    act_true = act_true[np.newaxis, :]

                    prob = Combiner.actor_step(Combiner.state, act_true).numpy()

                    a = utils.choose_action(prob, used)
                    used.append(a)

                    Combiner.step(x[:, a, :, :, :])

                print('epoch ', epoch, ' iteration ', iters, '  com_loss = ', Combiner.encoder_loss(y_true, Combiner.state))
                    

                if (iters % 50 == 0):
                    Combiner.validate(x, y, '{}_{}'.format(epoch, iters))
        
            Combiner.save_model()

    # 如果在最后一个训练阶段
    elif params.cur_step == 'fin':
        model = network.Fin(training = True)

        for epoch in range(params.epochs):
            iters = 0

            for x, y in dataset:
                iters += 1
                print('epoch ', epoch, ' iteration ', iters)

                model.step(x, y)

                if (iters % 200 == 0):
                    model.validate(x[0:1], y[0:1], '{}_{}'.format(epoch, iters))
        
            model.save_model()