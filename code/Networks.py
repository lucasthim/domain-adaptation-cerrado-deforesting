from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers


class Networks():
    def __init__(self, args):
        super(Networks, self).__init__()
        self.args = args
        self.base_number_of_features = 32

    def build_Unet_Arch(self, input_data, name="Unet_Arch"):
        with tf.variable_scope(name):
            # Encoder definition
            o_c1 = self.general_conv2d(input_data, self.base_number_of_features, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_1')
            o_mp1 = tf.layers.max_pooling2d(
                o_c1, 2, 2, name=name + '_maxpooling_1')
            o_c2 = self.general_conv2d(o_mp1, self.base_number_of_features * 2, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_2')
            o_mp2 = tf.layers.max_pooling2d(
                o_c2, 2, 2, name=name + '_maxpooling_2')
            o_c3 = self.general_conv2d(o_mp2, self.base_number_of_features * 4, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_3')
            o_mp3 = tf.layers.max_pooling2d(
                o_c3, 2, 2, name=name + '_maxpooling_3')
            o_c4 = self.general_conv2d(o_mp3, self.base_number_of_features * 8, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_4')
            o_mp4 = tf.layers.max_pooling2d(
                o_c4, 2, 2, name=name + '_maxpooling_4')
            o_c5 = self.general_conv2d(o_mp4, self.base_number_of_features * 16, 3, stride=1,
                                       padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_5')

            # Decoder definition
            o_d1 = self.general_deconv2d(o_c5, self.base_number_of_features * 8, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_1')
            o_me1 = tf.concat([o_d1, o_c4], 3)  # Skip conneco_c5tion
            o_d2 = self.general_deconv2d(o_me1, self.base_number_of_features * 4, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_2')
            o_me2 = tf.concat([o_d2, o_c3], 3)  # Skip connection
            o_d3 = self.general_deconv2d(o_me2, self.base_number_of_features * 2, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_3')
            o_me3 = tf.concat([o_d3, o_c2], 3)  # Skip connection
            o_d4 = self.general_deconv2d(o_me3, self.base_number_of_features, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_4')
            o_me4 = tf.concat([o_d4, o_c1], 3)  # Skip connection
            logits = tf.layers.conv2d(
                o_me4, self.args.num_classes, 1, 1, 'SAME', activation=None)
            prediction = tf.nn.softmax(logits, name=name + '_softmax')

            return logits, prediction, o_c5

    def build_Unet_Decoder_Arch(self, input_data, name="Unet_Decoder_Arch"):
        with tf.variable_scope(name):
            # Decoder definition
            o_d1 = self.general_deconv2d(input_data, self.base_number_of_features * 8, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_1')
            # o_me1 = tf.concat([o_d1, o_c4], 3) # Skip connection
            o_d2 = self.general_deconv2d(o_d1, self.base_number_of_features * 4, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_2')
            # o_me2 = tf.concat([o_d2, o_c3], 3) # Skip connection
            o_d3 = self.general_deconv2d(o_d2, self.base_number_of_features * 2, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_3')
            # o_me3 = tf.concat([o_d3, o_c2], 3) # Skip connection
            o_d4 = self.general_deconv2d(o_d3, self.base_number_of_features, 3, stride=2,
                                         padding='SAME', activation_function='relu', do_norm=False, name=name + '_deconv2d_4')
            # o_me4 = tf.concat([o_d4, o_c1], 3) # Skip connection
            logits = tf.layers.conv2d(
                o_d4, self.args.num_classes, 1, 1, 'SAME', activation=None)
            prediction = tf.nn.softmax(logits, name=name + '_softmax')

            return logits, prediction

    def build_DeepLab_Arch(self, inputs, is_training = False, name = "DeepLab_Arch", output_stride = 8, base_architecture='resnet_v2_101', 
                        pre_trained_model = './resnet_v2_101/resnet_v2_101.ckpt', batch_norm_decay = 0.9997, data_format='channels_last'):
        """Generator for DeepLab v3 plus models.

        Args:
        num_classes: The number of possible classes for image classification.
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
            the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        base_architecture: The architecture of base Resnet building block.
        pre_trained_model: The path to the directory that contains pre-trained models.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
        data_format: The input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
            Only 'channels_last' is supported currently.

        Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the DeepLab v3 model.
        """
        print('---------------------------------')
        print('Initializing DeepLab Architecture')
        print('---------------------------------')
        print('Input data shape:',inputs.shape)
        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
            raise ValueError(
                "'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")

        if base_architecture == 'resnet_v2_50':
            base_model = resnet_v2.resnet_v2_50
        else:
            base_model = resnet_v2.resnet_v2_101

        tf.logging.info('net shape: {}'.format(inputs.shape))
        # Resnet as Encoder
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            logits, end_points = base_model(inputs,
                                            num_classes=None,
                                            is_training=is_training,
                                            global_pool=False,
                                            output_stride=output_stride)

        if is_training:
            print('---------------------------------')
            print("Loading Pretrained ResNet V2 101")
            exclude = [base_architecture + '/logits', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(
                exclude=exclude)
            # print('Variables to restore:')
            # print({v.name.split(':')[0]: v for v in variables_to_restore})
            tf.train.init_from_checkpoint(pre_trained_model,
                                            {v.name.split(':')[0]: v for v in variables_to_restore})
            print("Loading Complete!")
            print('---------------------------------')

        inputs_size = tf.shape(inputs)[1:3]
        net = end_points[base_architecture + '/block4']
        encoder_output = atrous_spatial_pyramid_pooling(
            net, output_stride, batch_norm_decay, is_training)

        with tf.variable_scope("decoder"):
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    with tf.variable_scope("low_level_features"):
                        low_level_features = end_points[base_architecture +
                                                        '/block1/unit_3/bottleneck_v2/conv1']
                        low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                                [1, 1], stride=1, scope='conv_1x1')
                        low_level_features_size = tf.shape(
                            low_level_features)[1:3]

                    with tf.variable_scope("upsampling_logits"):
                        net = tf.image.resize_bilinear(
                            encoder_output, low_level_features_size, name='upsample_1')
                        net = tf.concat(
                            [net, low_level_features], axis=3, name='concat')
                        net = layers_lib.conv2d(
                            net, 256, [3, 3], stride=1, scope='conv_3x3_1')
                        net = layers_lib.conv2d(
                            net, 256, [3, 3], stride=1, scope='conv_3x3_2')
                        net = layers_lib.conv2d(net, self.args.num_classes, [
                                                1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
                        logits = tf.image.resize_bilinear(
                            net, inputs_size, name='upsample_2')
        # TODO: output logits from decoder
        prediction = tf.nn.softmax(logits, name = name + '_softmax')
        return prediction,logits,encoder_output
        # return logits

        # return model

    def build_DeepLab_Decoder_Arch(self, input_data, name="DeepLab_Decoder_Arch"):
        pass

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(
                input_data, filters, kernel_size, stride, padding, activation=None)

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv

    def general_deconv2d(self, input_data, filters=64, kernel_size=7, stride=1, stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="deconv2d"):
        with tf.variable_scope(name):
            deconv = tf.layers.conv2d_transpose(
                input_data, filters, kernel_size, (stride, stride), padding, activation=None)

            if do_norm:
                deconv = tf.layers.batch_normalization(deconv, momentum=0.9)

            if activation_function == "relu":
                deconv = tf.nn.relu(deconv, name='relu')
            if activation_function == "leakyrelu":
                deconv = tf.nn.leaky_relu(deconv, alpha=relu_factor)
            if activation_function == "elu":
                deconv = tf.nn.elu(deconv, name='elu')

            return deconv

    def atrous_spatial_pyramid_pooling(self, inputs, output_stride, batch_norm_decay, is_training, depth=256):
        """Atrous Spatial Pyramid Pooling.

        Args:
        inputs: A tensor of size [batch, height, width, channels].
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
            the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
        is_training: A boolean denoting whether the input is for training.
        depth: The depth of the ResNet unit output.

        Returns:
        The atrous spatial pyramid pooling output.
        """
        with tf.variable_scope("aspp"):
            if output_stride not in [8, 16]:
                raise ValueError('output_stride must be either 8 or 16.')

            atrous_rates = [6, 12, 18]
            if output_stride == 8:
                atrous_rates = [2*rate for rate in atrous_rates]

            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    inputs_size = tf.shape(inputs)[1:3]
                    # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                    # the rates are doubled when output stride = 8.
                    conv_1x1 = layers_lib.conv2d(
                        inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                    conv_3x3_1 = layers_lib.conv2d(
                        inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
                    conv_3x3_2 = layers_lib.conv2d(
                        inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
                    conv_3x3_3 = layers_lib.conv2d(
                        inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

                    # (b) the image-level features
                    with tf.variable_scope("image_level_features"):
                        # global average pooling
                        image_level_features = tf.reduce_mean(
                            inputs, [1, 2], name='global_average_pooling', keepdims=True)
                        # 1x1 convolution with 256 filters( and batch normalization)
                        image_level_features = layers_lib.conv2d(image_level_features, depth, [
                                                                1, 1], stride=1, scope='conv_1x1')
                        # bilinearly upsample features
                        image_level_features = tf.image.resize_bilinear(
                            image_level_features, inputs_size, name='upsample')

                    net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3,
                                    image_level_features], axis=3, name='concat')
                    net = layers_lib.conv2d(
                        net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

                    return net
