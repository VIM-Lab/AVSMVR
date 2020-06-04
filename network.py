import tensorflow as tf
import numpy as np
import PIL
import os

import params
import data
import utils


class Autoencoder:
    def __init__(self):
        self.voxel_encoder = self.make_voxel_encoder()
        self.generator = self.make_generator_model()
        self.loss_init()
        self.optimizer_init()
        self.save_model_init()


    def make_voxel_encoder(self):
        inputs = tf.keras.Input(shape = (params.voxel_resolution, params.voxel_resolution, params.voxel_resolution, 2))

        ve1 = tf.keras.layers.Conv3D(filters = 64, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(inputs) # shape = [16, 16, 16, filters]
        ve1 = tf.keras.layers.BatchNormalization()(ve1)
        ve1 = tf.keras.layers.LeakyReLU()(ve1)

        ve2 = tf.keras.layers.Conv3D(filters = 128, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(ve1) # shape = [8, 8, 8, filters]
        ve2 = tf.keras.layers.BatchNormalization()(ve2)
        ve2 = tf.keras.layers.LeakyReLU()(ve2)

        ve3 = tf.keras.layers.Conv3D(filters = 256, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(ve2) # shape = [4, 4, 4, filters]
        ve3 = tf.keras.layers.BatchNormalization()(ve3)
        ve3 = tf.keras.layers.LeakyReLU()(ve3)

        ve4 = tf.keras.layers.Flatten()(ve3) # shape = [4 * 4 * 4 * filters]
        outputs = tf.keras.layers.Dense(params.vector_size)(ve4) # shape = [params.vector_size]

        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model


    def make_generator_model(self):
        inputs = tf.keras.Input(shape = (params.vector_size,))
        
        # g0 = tf.keras.layers.Dense(1024)(inputs)
        g0 = tf.keras.layers.Reshape((4, 4, 4, 16))(inputs)

        g1 = tf.keras.layers.Conv3DTranspose(filters = 256, kernel_size = (3, 3, 3), strides = (1, 1, 1), padding = 'same')(g0) # shape = [4, 4, 4, filters]
        g1 = tf.keras.layers.BatchNormalization()(g1)
        g1 = tf.keras.layers.LeakyReLU()(g1)

        g2 = tf.keras.layers.Conv3DTranspose(filters = 128, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(g1) # shape = [8, 8, 8, filters]
        g2 = tf.keras.layers.BatchNormalization()(g2)
        g2 = tf.keras.layers.LeakyReLU()(g2)

        g3 = tf.keras.layers.Conv3DTranspose(filters = 64, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(g2) # shape = [16, 16, 16, filters]
        g3 = tf.keras.layers.BatchNormalization()(g3)
        g3 = tf.keras.layers.LeakyReLU()(g3)

        g4 = tf.keras.layers.Conv3DTranspose(filters = 2, kernel_size = (3, 3, 3), strides = (2, 2, 2), padding = 'same')(g3) # shape = [32, 32, 32, 2]
        g4 = tf.keras.activations.tanh(g4)

        outputs = tf.keras.layers.Softmax(axis = -1)(g4)

        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model


    def loss_init(self):    
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def optimizer_init(self):
        self.voxel_encoder_optimizer = tf.keras.optimizers.Adam()
        self.generator_optimizer = tf.keras.optimizers.Adam()


    def autoencoder_loss(self, real_voxel, fake_voxel):
        loss = self.cross_entropy(real_voxel, fake_voxel)
        return loss


    def save_model_init(self):
        self.checkpoint = tf.train.Checkpoint(voxel_encoder_optimizer = self.voxel_encoder_optimizer,
                                        generator_optimizer = self.generator_optimizer,
                                        voxel_encoder = self.voxel_encoder,
                                        generator = self.generator)
        self.manager = tf.train.CheckpointManager(checkpoint = self.checkpoint, directory = params.checkpoint_path, max_to_keep = 5)


    def save_model(self):
        self.manager.save()


    def load_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(params.checkpoint_path))


    # 一步迭代
    @tf.function
    def step(self, real_voxel):
        with tf.GradientTape() as voxel_enc_tape, tf.GradientTape() as gen_tape:
            noise = self.voxel_encoder(real_voxel, training = True)
            fake_voxel = self.generator(noise, training = True)

            ae_loss = self.autoencoder_loss(real_voxel, fake_voxel)

        tf.print('ae_loss = ', ae_loss)

        voxel_enc_grad = voxel_enc_tape.gradient(ae_loss, self.voxel_encoder.trainable_variables)
        gen_grad = gen_tape.gradient(ae_loss, self.generator.trainable_variables)

        self.voxel_encoder_optimizer.apply_gradients(zip(voxel_enc_grad, self.voxel_encoder.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))


    # 生成并保存可视化结果
    def validate(self, real_voxel, name):
        noise = self.voxel_encoder(real_voxel, training = False)
        voxel = self.generator(noise, training = False)

        voxel = utils.dicide_voxel(voxel)
        utils.save_voxel(voxel, name)



class Combiner:
    def __init__(self, training):
        self.create_autoencoder()
        self.image_encoder = self.make_image_encoder()
        self.actor = self.make_actor()
        self.loss_init()
        self.optimizer_init()
        self.save_model_init()
        if training == True:
            self.load_autoencoder()
        elif training == False:
            self.load_model()


    def load_autoencoder(self):
        autoencoder = Autoencoder()
        autoencoder.load_model()
        self.voxel_encoder_optimizer = autoencoder.voxel_encoder_optimizer
        self.generator_optimizer = autoencoder.generator_optimizer
        self.voxel_encoder = autoencoder.voxel_encoder
        self.generator = autoencoder.generator


    def create_autoencoder(self):
        autoencoder = Autoencoder()
        self.voxel_encoder = autoencoder.make_voxel_encoder()
        self.generator = autoencoder.make_generator_model()
        self.voxel_encoder_optimizer = autoencoder.voxel_encoder_optimizer
        self.generator_optimizer = autoencoder.generator_optimizer

    def make_image_encoder(self):
        inputs = tf.keras.Input(shape = (params.image_size, params.image_size, 4)) # shape = [128, 128, 4]

        e1 = tf.keras.layers.Conv2D(filters = 4, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(inputs)
        e1 = tf.keras.layers.BatchNormalization()(e1)
        e1 = tf.keras.layers.LeakyReLU()(e1)
        e1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e1)
        e1 = tf.keras.layers.BatchNormalization()(e1)
        e1 = tf.keras.layers.LeakyReLU()(e1)
        
        s1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same')(inputs)
        e1 = e1 + s1
        e1 = tf.keras.layers.MaxPool2D()(e1) # shape = [64, 64, 64]


        e2 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e1)
        e2 = tf.keras.layers.BatchNormalization()(e2)
        e2 = tf.keras.layers.LeakyReLU()(e2)
        e2 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e2)
        e2 = tf.keras.layers.BatchNormalization()(e2)
        e2 = tf.keras.layers.LeakyReLU()(e2)

        s2 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (1, 1), strides = (1, 1), padding = 'same')(e1)
        e2 = e2 + s2
        e2 = tf.keras.layers.MaxPool2D()(e2) # shape = [32, 32, 128]


        e3 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e2)
        e3 = tf.keras.layers.BatchNormalization()(e3)
        e3 = tf.keras.layers.LeakyReLU()(e3)
        e3 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e3)
        e3 = tf.keras.layers.BatchNormalization()(e3)
        e3 = tf.keras.layers.LeakyReLU()(e3)

        s3 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'same')(e2)
        e3 = e3 + s3
        e3 = tf.keras.layers.MaxPool2D()(e3) # shape = [16, 16, 256]


        e4 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e3)
        e4 = tf.keras.layers.BatchNormalization()(e4)
        e4 = tf.keras.layers.LeakyReLU()(e4)
        e4 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e4)
        e4 = tf.keras.layers.BatchNormalization()(e4)
        e4 = tf.keras.layers.LeakyReLU()(e4)

        s4 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'same')(e3)
        e4 = e4 + s4
        e4 = tf.keras.layers.MaxPool2D()(e4) # shape = [8, 8, 512]


        e5 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e4)
        e5 = tf.keras.layers.BatchNormalization()(e5)
        e5 = tf.keras.layers.LeakyReLU()(e5)
        e5 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = 'same')(e5)
        e5 = tf.keras.layers.BatchNormalization()(e5)
        e5 = tf.keras.layers.LeakyReLU()(e5)

        s5 = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), strides = (1, 1), padding = 'same')(e4)
        e5 = e5 + s5
        e5 = tf.keras.layers.MaxPool2D()(e5) # shape = [4, 4, 512]


        e6 = tf.keras.layers.Flatten()(e5)
        e6 = tf.keras.layers.Dense(1024)(e6)

        outputs = e6

        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model


    def make_actor(self):
        inputs = tf.keras.Input(shape = (params.vector_size,))

        a1 = tf.keras.layers.Dense(1024)(inputs) # shape = [1024]
        a1 = tf.keras.layers.BatchNormalization()(a1)
        a1 = tf.keras.layers.LeakyReLU()(a1)

        a2 = tf.keras.layers.Dense(24)(a1) # shape = [24]
        a2 = tf.keras.layers.BatchNormalization()(a2)
        a2 = tf.keras.layers.LeakyReLU()(a2)

        outputs = tf.keras.layers.Softmax(axis = -1)(a2) # shape = [24]

        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model


    def loss_init(self):    
        self.mean_squared = tf.keras.losses.MeanSquaredError()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def encoder_loss(self, y_true, y_pred):
        loss = self.mean_squared(y_true, y_pred)
        return loss


    def actor_loss(self, y_true, y_pred):
        loss = self.cross_entropy(y_true, y_pred)
        return loss

    
    def optimizer_init(self):
        self.image_encoder_optimizer = tf.keras.optimizers.Adam()
        self.actor_optimizer = tf.keras.optimizers.Adam()


    def save_model_init(self):
        self.checkpoint = tf.train.Checkpoint(voxel_encoder_optimizer = self.voxel_encoder_optimizer,
                                            generator_optimizer = self.generator_optimizer,
                                            voxel_encoder = self.voxel_encoder,
                                            generator = self.generator,
            
                                            image_encoder_optimizer = self.image_encoder_optimizer,
                                            actor_optimizer = self.actor_optimizer,
                                            image_encoder = self.image_encoder,
                                            actor = self.actor)
        self.manager = tf.train.CheckpointManager(checkpoint = self.checkpoint, directory = params.checkpoint_path, max_to_keep = 5)


    def save_model(self):
        self.manager.save()


    def load_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(params.checkpoint_path))
        

    # 初始化state
    def init_step(self, image, y_true):
        with tf.GradientTape() as image_enc_tape:
            self.state = self.image_encoder(image, training = True)

            com_loss = self.encoder_loss(y_true, self.state)

        image_enc_grad = image_enc_tape.gradient(com_loss, self.image_encoder.trainable_variables)

        self.image_encoder_optimizer.apply_gradients(zip(image_enc_grad, self.image_encoder.trainable_variables))

    # image_encoder迭代
    def encoder_step(self, image, y_true):
        with tf.GradientTape() as image_enc_tape:
            append_state = self.image_encoder(image, training = True) # shape = [batch_size, 1024]

            self.state = tf.reduce_max(tf.stack([self.state, append_state], axis = 1), axis = 1) # 先将两个tensor堆叠，再以最大值去掉一维，相当于在某个维度上求max

            com_loss = self.encoder_loss(y_true, self.state)

        image_enc_grad = image_enc_tape.gradient(com_loss, self.image_encoder.trainable_variables)
        self.image_encoder_optimizer.apply_gradients(zip(image_enc_grad, self.image_encoder.trainable_variables))

        return com_loss


    # actor迭代
    def actor_step(self, s, act_true):
        with tf.GradientTape() as act_tape:
            prob = self.actor(s, training = True)

            act_loss = self.actor_loss(act_true, prob)

        actor_grad = act_tape.gradient(act_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        return prob


    # 当前状态前进一步
    def step(self, image):
        append_state = self.image_encoder(image, training = False)
        self.state = tf.reduce_max(tf.stack([self.state, append_state], axis = 1), axis = 1)


    # 生成并保存可视化结果
    def validate(self, image, y, name):
        # temp = tf.map_fn(lambda x : self.image_encoder(x, training = False), image)

        self.state = self.image_encoder(image[:, 0, :, :, :], training = False) # shape = [batch_size, 1024]
        used = [0]
        
        for _t in range(params.time_steps - 1):
            prob = self.actor(self.state).numpy()

            action = utils.choose_action(prob, used)
            used.append(action)

            append_state = self.image_encoder(image[:, action, :, :, :], training = False)
            self.state = tf.reduce_max(tf.stack([self.state, append_state], axis = 1), axis = 1)

        voxel = self.generator(self.state, training = False)

        voxel = utils.dicide_voxel(voxel)
        utils.save_voxel(voxel, '{}_pridict'.format(name))
        utils.save_voxel(y, '{}_true'.format(name))

        y = y[0]
        y = np.argmax(y, -1)
        voxel = np.argmax(voxel, -1)

        iou = utils.cal_iou(y, voxel)
        print('iou = ', iou)



class Fin:
    def __init__(self, training):
        if training == True:
            combiner = Combiner(training = False)
        elif training == False:
            combiner = Combiner(training = None)

        self.voxel_encoder = combiner.voxel_encoder
        self.generator = combiner.generator
        self.image_encoder = combiner.image_encoder
        self.actor = combiner.actor

        self.loss_init()
        self.optimizer_init()
        self.save_model_init()


    def loss_init(self):    
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def optimizer_init(self):
        self.voxel_encoder_optimizer = tf.keras.optimizers.Adam()
        self.generator_optimizer = tf.keras.optimizers.Adam()
        self.image_encoder_optimizer = tf.keras.optimizers.Adam()
        self.actor_optimizer = tf.keras.optimizers.Adam()


    def save_model_init(self):
        self.checkpoint = tf.train.Checkpoint(voxel_encoder_optimizer = self.voxel_encoder_optimizer,
                                            generator_optimizer = self.generator_optimizer,
                                            voxel_encoder = self.voxel_encoder,
                                            generator = self.generator,
            
                                            image_encoder_optimizer = self.image_encoder_optimizer,
                                            actor_optimizer = self.actor_optimizer,
                                            image_encoder = self.image_encoder,
                                            actor = self.actor)
        self.manager = tf.train.CheckpointManager(checkpoint = self.checkpoint, directory = params.checkpoint_path, max_to_keep = 5)


    def save_model(self):
        self.manager.save()


    def load_model(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(params.checkpoint_path))


    def fin_loss(self, real_voxel, fake_voxel):
        loss = self.cross_entropy(real_voxel, fake_voxel)
        return loss


    def step(self, image, real_voxel):
        with tf.GradientTape() as image_enc_tape, tf.GradientTape() as gen_tape:
            self.state = self.image_encoder(image[:, 0, :, :, :], training = True)
            used= [0]

            for _t in range(params.time_steps - 1):
                prob = self.actor(self.state).numpy()

                action = utils.choose_action(prob, used)
                used.append(action)

                append_state = self.image_encoder(image[:, action, :, :, :], training = True)
                self.state = tf.reduce_max(tf.stack([self.state, append_state], axis = 1), axis = 1)

            voxel = self.generator(self.state, training = True)

            fin_loss = self.fin_loss(real_voxel, voxel)

        tf.print('fin_loss = ', fin_loss)

        image_enc_grad = image_enc_tape.gradient(fin_loss, self.image_encoder.trainable_variables)
        gen_grad = gen_tape.gradient(fin_loss, self.generator.trainable_variables)

        self.image_encoder_optimizer.apply_gradients(zip(image_enc_grad, self.image_encoder.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))


    # 生成并保存可视化结果
    def validate(self, image, y, name):
        # temp = tf.map_fn(lambda x : self.image_encoder(x, training = False), image)

        self.state = self.image_encoder(image[:, 0, :, :, :], training = False) # shape = [batch_size, 1024]
        used = [0]
        
        for _t in range(params.time_steps - 1):
            prob = self.actor(self.state).numpy()

            action = utils.choose_action(prob, used)
            used.append(action)

            append_state = self.image_encoder(image[:, action, :, :, :], training = False)
            self.state = tf.reduce_max(tf.stack([self.state, append_state], axis = 1), axis = 1)

        voxel = self.generator(self.state, training = False)

        voxel = utils.dicide_voxel(voxel)
        utils.save_voxel(voxel, '{}_pridict'.format(name))
        utils.save_voxel(y, '{}_true'.format(name))

        y = y[0]
        y = np.argmax(y, -1)
        voxel = np.argmax(voxel, -1)

        iou = utils.cal_iou(y, voxel)
        print('iou = ', iou)


