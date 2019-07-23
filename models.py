import tensorflow as tf
from tensorflow.python.ops.init_ops import he_normal as he


def _seresblock_proj(name, input, weight1, weight2, weight_proj, filter_num, ratio=4):
    with tf.variable_scope(name):
        fc1_w = tf.get_variable(name=name + '_fc1_w', shape=[filter_num, int(filter_num / ratio)], initializer=he())
        fc2_w = tf.get_variable(name=name + '_fc2_w', shape=[int(filter_num / ratio), filter_num], initializer=he())

        # residual block
        conv = tf.nn.conv2d(input, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.layers.batch_normalization(conv)
        af = tf.nn.relu(bn)
        conv = tf.nn.conv2d(af, weight2, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.layers.batch_normalization(conv)
        proj = tf.nn.conv2d(input, weight_proj, strides=(1, 1, 1, 1), padding='SAME')
        bn_proj = tf.layers.batch_normalization(proj)

        # squeeze & exitation block
        gap = tf.reduce_mean(bn_proj, axis=[1, 2])  # (B x 1 x C)
        fc1 = tf.reshape(gap, shape=[-1, filter_num])  # (B x C)
        fc1 = tf.matmul(fc1, fc1_w)  # (B x C/R)
        relu = tf.nn.relu(fc1)  # (B x C/R)
        fc2 = tf.matmul(relu, fc2_w)  # (B x C)
        sig = tf.nn.sigmoid(fc2)  # (B x C)
        descriptor = tf.reshape(sig, shape=[-1, 1, 1, filter_num])

        out = tf.nn.relu(bn + bn_proj * descriptor)

    return out


def _seresblock_double(name, input, weight1, weight2, filter_num, ratio=4):
    with tf.variable_scope(name):
        fc1_w = tf.get_variable(name=name + '_fc1_w', shape=[filter_num, int(filter_num / ratio)], initializer=he())
        fc2_w = tf.get_variable(name=name + '_fc2_w', shape=[int(filter_num / ratio), filter_num], initializer=he())

        # residual block
        bn1 = tf.layers.batch_normalization(input)
        relu1 = tf.nn.leaky_relu(bn1)
        conv1 = tf.nn.conv2d(relu1, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn2 = tf.layers.batch_normalization(conv1)
        relu2 = tf.nn.leaky_relu(bn2)
        conv2 = tf.nn.conv2d(relu2, weight2, strides=(1, 1, 1, 1), padding='SAME')

        # squeeze & exitation block
        gap = tf.reduce_mean(conv2, axis=[1, 2])  # (B x 1 x C)
        fc1 = tf.reshape(gap, shape=[-1, filter_num])  # (B x C)
        fc1 = tf.matmul(fc1, fc1_w)  # (B x C/R)
        relu = tf.nn.relu(fc1)  # (B x C/R)
        fc2 = tf.matmul(relu, fc2_w)  # (B x C)
        sig = tf.nn.sigmoid(fc2)  # (B x C)
        descriptor = tf.reshape(sig, shape=[-1, 1, 1, filter_num])

        out = input + conv2 * descriptor

    return out


def _resblock_proj(name, input, weight1, weight2, weight_proj):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(input, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.layers.batch_normalization(conv)
        af = tf.nn.relu(bn)
        conv = tf.nn.conv2d(af, weight2, strides=(1, 1, 1, 1), padding='SAME')
        bn = tf.layers.batch_normalization(conv)
        proj = tf.nn.conv2d(input, weight_proj, strides=(1, 1, 1, 1), padding='SAME')
        bn_proj = tf.layers.batch_normalization(proj)
        out = tf.nn.relu(bn + bn_proj)

    return out


# full pre-activation block from paper 'Identity Mapping in Deep Residual Networks (2016)'
def _resblock_double(name, input, weight1, weight2):
    with tf.variable_scope(name):
        bn1 = tf.layers.batch_normalization(input)
        relu1 = tf.nn.leaky_relu(bn1)
        conv1 = tf.nn.conv2d(relu1, weight1, strides=(1, 1, 1, 1), padding='SAME')
        bn2 = tf.layers.batch_normalization(conv1)
        relu2 = tf.nn.leaky_relu(bn2)
        conv2 = tf.nn.conv2d(relu2, weight2, strides=(1, 1, 1, 1), padding='SAME')

    return conv2 + input


class SeResNetDouble4_448:
    def __init__(self, name, images, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.conv_w = tf.get_variable(name='cls_conv_w', shape=[3, 3, channel, 64], initializer=he())

            self.res_w1_1 = tf.get_variable(name='cls_res_w1_1', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='cls_res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_3 = tf.get_variable(name='cls_res_w1_3', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_4 = tf.get_variable(name='cls_res_w1_4', shape=[3, 3, 64, 64], initializer=he())

            self.proj_w2 = tf.get_variable(name='cls_proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='cls_res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='cls_res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_3 = tf.get_variable(name='cls_res_w2_3', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_4 = tf.get_variable(name='cls_res_w2_4', shape=[3, 3, 128, 128], initializer=he())

            self.proj_w3 = tf.get_variable(name='cls_proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3_1 = tf.get_variable(name='cls_res_w3_1', shape=[3, 3, 128, 256], initializer=he())
            self.res_w3_2 = tf.get_variable(name='cls_res_w3_2', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_3 = tf.get_variable(name='cls_res_w3_3', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_4 = tf.get_variable(name='cls_res_w3_4', shape=[3, 3, 256, 256], initializer=he())

            self.proj_w4 = tf.get_variable(name='cls_proj_w4', shape=[1, 1, 256, 320], initializer=he())
            self.res_w4_1 = tf.get_variable(name='cls_res_w4_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w4_2 = tf.get_variable(name='cls_res_w4_2', shape=[3, 3, 320, 320], initializer=he())
            self.res_w4_3 = tf.get_variable(name='cls_res_w4_3', shape=[3, 3, 320, 320], initializer=he())
            self.res_w4_4 = tf.get_variable(name='cls_res_w4_4', shape=[3, 3, 320, 320], initializer=he())

            self.conv_last_w1 = tf.get_variable(name='cls_conv_last_w1', shape=[3, 3, 320, 512], initializer=he())
            self.fc_w1 = tf.get_variable(name='fc_w1', shape=[2048, classes])

            # 2 - Graphs
            self.conv = tf.nn.relu(tf.nn.conv2d(images, self.conv_w, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool = tf.nn.max_pool(self.conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,224,224,64)

            self.res1_1 = _seresblock_double('cls_res1_1', self.pool, self.res_w1_1, self.res_w1_2, 64)
            self.res1_2 = _seresblock_double('cls_res1_2', self.res1_1, self.res_w1_3, self.res_w1_4, 64)
            self.pool1 = tf.nn.max_pool(self.res1_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,112,112,64)

            self.res2_1 = _seresblock_proj('cls_res2_1', self.pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.res2_2 = _seresblock_double('cls_res2_2', self.res2_1, self.res_w2_3, self.res_w2_4, 128)
            self.pool2 = tf.nn.max_pool(self.res2_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,56,56,128)

            self.res3_1 = _seresblock_proj('cls_res3_1', self.pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 256)
            self.res3_2 = _seresblock_double('cls_res3_2', self.res3_1, self.res_w3_3, self.res_w3_4, 256)
            self.pool3 = tf.nn.max_pool(self.res3_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,28,28,256)

            self.res4_1 = _seresblock_proj('cls_res4_1', self.pool3, self.res_w4_1, self.res_w4_2, self.proj_w4, 320)
            self.res4_2 = _seresblock_double('cls_res4_2', self.res4_1, self.res_w4_3, self.res_w4_4, 320)
            self.pool4 = tf.nn.max_pool(self.res4_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,14,14,320)

            self.conv_last = tf.nn.relu(tf.nn.conv2d(self.pool4, self.conv_last_w1, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool_last = tf.nn.avg_pool(self.conv_last, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')

            self.flat = tf.reshape(self.pool_last, shape=[-1, 2048])
            self.logits = tf.matmul(self.flat, self.fc_w1)

    def get_cam(self, width, height):
        with tf.variable_scope('cam'):
            conv_resized = tf.image.resize_bilinear(self.conv_last, size=(height, width))  # 128x128x512
            conv_flat = tf.reshape(conv_resized, shape=[-1, height * width, 512])
            gap = tf.nn.avg_pool(self.conv_last, ksize=(1, 14, 14, 1), strides=(1, 14, 14, 1), padding='SAME')
            gap_flat = tf.reshape(gap, shape=[-1, 512, 1])
            cam = tf.reshape(tf.matmul(conv_flat, gap_flat), shape=[-1, height, width, 1])

        return cam


class ResNetDouble4_448:
    def __init__(self, name, images, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.conv_w = tf.get_variable(name='cls_conv_w', shape=[3, 3, channel, 64], initializer=he())

            self.res_w1_1 = tf.get_variable(name='cls_res_w1_1', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='cls_res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_3 = tf.get_variable(name='cls_res_w1_3', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_4 = tf.get_variable(name='cls_res_w1_4', shape=[3, 3, 64, 64], initializer=he())

            self.proj_w2 = tf.get_variable(name='cls_proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='cls_res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='cls_res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_3 = tf.get_variable(name='cls_res_w2_3', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_4 = tf.get_variable(name='cls_res_w2_4', shape=[3, 3, 128, 128], initializer=he())

            self.proj_w3 = tf.get_variable(name='cls_proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3_1 = tf.get_variable(name='cls_res_w3_1', shape=[3, 3, 128, 256], initializer=he())
            self.res_w3_2 = tf.get_variable(name='cls_res_w3_2', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_3 = tf.get_variable(name='cls_res_w3_3', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_4 = tf.get_variable(name='cls_res_w3_4', shape=[3, 3, 256, 256], initializer=he())

            self.proj_w4 = tf.get_variable(name='cls_proj_w4', shape=[1, 1, 256, 320], initializer=he())
            self.res_w4_1 = tf.get_variable(name='cls_res_w4_1', shape=[3, 3, 256, 320], initializer=he())
            self.res_w4_2 = tf.get_variable(name='cls_res_w4_2', shape=[3, 3, 320, 320], initializer=he())
            self.res_w4_3 = tf.get_variable(name='cls_res_w4_3', shape=[3, 3, 320, 320], initializer=he())
            self.res_w4_4 = tf.get_variable(name='cls_res_w4_4', shape=[3, 3, 320, 320], initializer=he())

            self.conv_last_w1 = tf.get_variable(name='conv_last_w1', shape=[3, 3, 320, 512], initializer=he())
            self.fc_w1 = tf.get_variable(name='fc_w1', shape=[2048, classes])

            # 2 - Graphs
            self.conv = tf.nn.relu(tf.nn.conv2d(images, self.conv_w, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool = tf.nn.max_pool(self.conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,64,64,64)

            self.res1_1 = _resblock_double('cls_res1_1', self.pool, self.res_w1_1, self.res_w1_2)
            self.res1_2 = _resblock_double('cls_res1_2', self.res1_1, self.res_w1_3, self.res_w1_4)
            self.pool1 = tf.nn.max_pool(self.res1_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,32,32,64)

            self.res2_1 = _resblock_proj('cls_res2_1', self.pool1, self.res_w2_1, self.res_w2_2, self.proj_w2)
            self.res2_2 = _resblock_double('cls_res2_2', self.res2_1, self.res_w2_3, self.res_w2_4)
            self.pool2 = tf.nn.max_pool(self.res2_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,16,16,128)

            self.res3_1 = _resblock_proj('cls_res3_1', self.pool2, self.res_w3_1, self.res_w3_2, self.proj_w3)
            self.res3_2 = _resblock_double('cls_res3_2', self.res3_1, self.res_w3_3, self.res_w3_4)
            self.pool3 = tf.nn.max_pool(self.res3_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,8,8,256)

            self.res4_1 = _resblock_proj('cls_res4_1', self.pool3, self.res_w4_1, self.res_w4_2, self.proj_w4)
            self.res4_2 = _resblock_double('cls_res4_2', self.res4_1, self.res_w4_3, self.res_w4_4)
            self.pool4 = tf.nn.max_pool(self.res4_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,8,8,256)

            self.conv_last = tf.nn.relu(tf.nn.conv2d(self.pool4, self.conv_last_w1, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool_last = tf.nn.avg_pool(self.conv_last, ksize=(1, 7, 7, 1), strides=(1, 7, 7, 1), padding='SAME')

            self.flat = tf.reshape(self.pool_last, shape=[-1, 2048])
            self.logits = tf.matmul(self.flat, self.fc_w1)


class SeResNetDouble3_128:
    def __init__(self, name, images, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.conv_w = tf.get_variable(name='cls_conv_w', shape=[3, 3, channel, 64], initializer=he())

            self.res_w1_1 = tf.get_variable(name='cls_res_w1_1', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='cls_res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_3 = tf.get_variable(name='cls_res_w1_3', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_4 = tf.get_variable(name='cls_res_w1_4', shape=[3, 3, 64, 64], initializer=he())

            self.proj_w2 = tf.get_variable(name='cls_proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='cls_res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='cls_res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_3 = tf.get_variable(name='cls_res_w2_3', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_4 = tf.get_variable(name='cls_res_w2_4', shape=[3, 3, 128, 128], initializer=he())

            self.proj_w3 = tf.get_variable(name='cls_proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3_1 = tf.get_variable(name='cls_res_w3_1', shape=[3, 3, 128, 256], initializer=he())
            self.res_w3_2 = tf.get_variable(name='cls_res_w3_2', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_3 = tf.get_variable(name='cls_res_w3_3', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_4 = tf.get_variable(name='cls_res_w3_4', shape=[3, 3, 256, 256], initializer=he())

            self.conv_last_w1 = tf.get_variable(name='cls_conv_last_w1', shape=[3, 3, 256, 512], initializer=he())
            self.fc_w1 = tf.get_variable(name='fc_w1', shape=[2048, classes])  # bounding box coords
            # self.concat_w = tf.get_variable(name='concat_w', shape=[3072, classes])

            # 2 - Graphs
            self.conv = tf.nn.relu(tf.nn.conv2d(images, self.conv_w, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool = tf.nn.max_pool(self.conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,64,64,64)

            self.res1_1 = _seresblock_double('cls_res1_1', self.pool, self.res_w1_1, self.res_w1_2, 64)
            self.res1_2 = _seresblock_double('cls_res1_2', self.res1_1, self.res_w1_3, self.res_w1_4, 64)
            self.pool1 = tf.nn.max_pool(self.res1_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,32,32,64)

            self.res2_1 = _seresblock_proj('cls_res2_1', self.pool1, self.res_w2_1, self.res_w2_2, self.proj_w2, 128)
            self.res2_2 = _seresblock_double('cls_res2_2', self.res2_1, self.res_w2_3, self.res_w2_4, 128)
            self.pool2 = tf.nn.max_pool(self.res2_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,16,16,128)

            self.res3_1 = _seresblock_proj('cls_res3_1', self.pool2, self.res_w3_1, self.res_w3_2, self.proj_w3, 256)
            self.res3_2 = _seresblock_double('cls_res3_2', self.res3_1, self.res_w3_3, self.res_w3_4, 256)
            self.pool3 = tf.nn.max_pool(self.res3_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,8,8,256)

            self.conv_last = tf.nn.relu(tf.nn.conv2d(self.pool3, self.conv_last_w1, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool_last = tf.nn.avg_pool(self.conv_last, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')

            self.flat = tf.reshape(self.pool_last, shape=[-1, 2048])
            self.logits = tf.matmul(self.flat, self.fc_w1)  # (?, classes * 4)

            # self.reg_fc_graph = tf.reshape(reg_fc_graph, [-1, 1024])
            # self.concat_flat = tf.concat([self.flat, self.reg_fc_graph], axis=1)
            # self.concat_logits = tf.matmul(self.concat_flat, self.concat_w)


# 3-layered ResNet for (128, 128) images
class ResNetDouble3_128:
    def __init__(self, name, images, channel, classes):
        with tf.variable_scope(name):
            # 1 - Filters
            self.conv_w = tf.get_variable(name='cls_conv_w', shape=[3, 3, channel, 64], initializer=he())

            self.res_w1_1 = tf.get_variable(name='cls_res_w1_1', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_2 = tf.get_variable(name='cls_res_w1_2', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_3 = tf.get_variable(name='cls_res_w1_3', shape=[3, 3, 64, 64], initializer=he())
            self.res_w1_4 = tf.get_variable(name='cls_res_w1_4', shape=[3, 3, 64, 64], initializer=he())

            self.proj_w2 = tf.get_variable(name='cls_proj_w2', shape=[1, 1, 64, 128], initializer=he())
            self.res_w2_1 = tf.get_variable(name='cls_res_w2_1', shape=[3, 3, 64, 128], initializer=he())
            self.res_w2_2 = tf.get_variable(name='cls_res_w2_2', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_3 = tf.get_variable(name='cls_res_w2_3', shape=[3, 3, 128, 128], initializer=he())
            self.res_w2_4 = tf.get_variable(name='cls_res_w2_4', shape=[3, 3, 128, 128], initializer=he())

            self.proj_w3 = tf.get_variable(name='cls_proj_w3', shape=[1, 1, 128, 256], initializer=he())
            self.res_w3_1 = tf.get_variable(name='cls_res_w3_1', shape=[3, 3, 128, 256], initializer=he())
            self.res_w3_2 = tf.get_variable(name='cls_res_w3_2', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_3 = tf.get_variable(name='cls_res_w3_3', shape=[3, 3, 256, 256], initializer=he())
            self.res_w3_4 = tf.get_variable(name='cls_res_w3_4', shape=[3, 3, 256, 256], initializer=he())

            self.conv_last_w1 = tf.get_variable(name='conv_last_w1', shape=[3, 3, 256, 512], initializer=he())
            self.fc_w1 = tf.get_variable(name='fc_w1', shape=[2048, classes])  # bounding box coords
            # self.concat_w = tf.get_variable(name='concat_w', shape=[3072, classes])

            # 2 - Graphs
            self.conv = tf.nn.relu(tf.nn.conv2d(images, self.conv_w, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool = tf.nn.max_pool(self.conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,64,64,64)

            self.res1_1 = _resblock_double('cls_res1_1', self.pool, self.res_w1_1, self.res_w1_2)
            self.res1_2 = _resblock_double('cls_res1_2', self.res1_1, self.res_w1_3, self.res_w1_4)
            self.pool1 = tf.nn.max_pool(self.res1_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,32,32,64)

            self.res2_1 = _resblock_proj('cls_res2_1', self.pool1, self.res_w2_1, self.res_w2_2, self.proj_w2)
            self.res2_2 = _resblock_double('cls_res2_2', self.res2_1, self.res_w2_3, self.res_w2_4)
            self.pool2 = tf.nn.max_pool(self.res2_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,16,16,128)

            self.res3_1 = _resblock_proj('cls_res3_1', self.pool2, self.res_w3_1, self.res_w3_2, self.proj_w3)
            self.res3_2 = _resblock_double('cls_res3_2', self.res3_1, self.res_w3_3, self.res_w3_4)
            self.pool3 = tf.nn.max_pool(self.res3_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')  # (?,8,8,256)

            self.conv_last = tf.nn.relu(tf.nn.conv2d(self.pool3, self.conv_last_w1, strides=(1, 1, 1, 1), padding='SAME'))
            self.pool_last = tf.nn.avg_pool(self.conv_last, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')

            self.flat = tf.reshape(self.pool_last, shape=[-1, 2048])
            self.logits = tf.matmul(self.flat, self.fc_w1)  # (?, classes * 4)

            # self.reg_fc_graph = tf.reshape(reg_fc_graph, [-1, 1024])
            # self.concat_flat = tf.concat([self.flat, self.reg_fc_graph], axis=1)
            # self.concat_logits = tf.matmul(self.concat_flat, self.concat_w)


class PatchProcessor:
    def __init__(self, name, images, bb, ori_w=448, ori_h=448, new_w=128, new_h=128, batch_size=32):
        """
        Description:
            initialize model for sinus cropping into given size

        Args:
            name: custom model name
            images: batch of X-ray images with (?, 448, 448, 1) dim
            bb: batch of bounding box central coords and size info with [x, y, w, h] form and (?, 4) dim
            w, h: after cropping, resize images into given width and height

        Funtions:
            _center_to_yxyx : change xywh coordinates to yxyx
            random_distortion : translate, rotate, adjust contrast and flip randomly after sinus cropping

        Main Tensors:
            cropped_img : cropped sinus images with dimension (?, h, w, 1)
        """
        with tf.variable_scope(name):
            # B2 - Crop images by regression results
            self.reg_logits = bb
            self.lt_maxil_boxes = self._center_to_yxyx(0, ori_w, ori_h)
            self.lt_maxil_ind = tf.to_int32(tf.linspace(0., batch_size - 1, batch_size))  # tensor with 0 ~ 31
            self.lt_maxil = tf.image.crop_and_resize(images, self.lt_maxil_boxes, self.lt_maxil_ind, (new_h, new_w))
            self.rt_maxil_boxes = self._center_to_yxyx(1, ori_w, ori_h)
            self.rt_maxil_ind = tf.to_int32(tf.linspace(0., batch_size - 1, batch_size))  # tensor with 0 ~ 31
            self.rt_maxil = tf.image.crop_and_resize(images, self.rt_maxil_boxes, self.rt_maxil_ind, (new_h, new_w))
            self.rt_maxil = tf.image.flip_left_right(self.rt_maxil)
            self.cropped_imgs = tf.concat([self.lt_maxil, self.rt_maxil], axis=0)  # stacked cropped images (2?,56,56,1)

    def _center_to_yxyx(self, classes, ori_w, ori_h):
        with tf.variable_scope(str(classes)):
            x = (self.reg_logits[:, 4 * classes + 0:4 * classes + 1])  # (?, 1)
            y = (self.reg_logits[:, 4 * classes + 1:4 * classes + 2])  # (?, 1)
            w = (self.reg_logits[:, 4 * classes + 2:4 * classes + 3])  # (?, 1)
            h = (self.reg_logits[:, 4 * classes + 3:4 * classes + 4])  # (?, 1)

            y1, x1 = (y - h / 2) / ori_h, (x - w / 2) / ori_w
            y2, x2 = (y + h / 2) / ori_h, (x + w / 2) / ori_w
            coords = tf.concat([y1, x1, y2, x2], axis=1)

            return coords

    def random_distortion(self, batch_size, do_translate=False, do_rotate=False, do_contrast=False, do_flip=False,
                          trsl_pixel=15, rot_angle=5, cont_ratio=0.5):
        if do_translate:
            rand_translation = tf.random_uniform([batch_size, 2], -1 * trsl_pixel, trsl_pixel, dtype=tf.float32)
            self.cropped_imgs = tf.contrib.image.translate(self.cropped_imgs, rand_translation, 'BILINEAR')
        if do_rotate:
            rand_rotation = tf.random_uniform([batch_size], -1 * rot_angle, rot_angle, dtype=tf.float32)
            self.cropped_imgs = tf.contrib.image.rotate(self.cropped_imgs, rand_rotation)
        if do_contrast:
            self.cropped_imgs = tf.image.random_contrast(self.cropped_imgs, 1 - cont_ratio, 1 + cont_ratio)
        if do_flip:
            self.cropped_imgs = tf.image.random_flip_left_right(self.cropped_imgs)