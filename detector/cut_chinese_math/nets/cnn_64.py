import tensorflow as tf
import tensorflow.contrib.slim as slim
from detector.cut_chinese_math.cut_config import Config as config

class Model():
    def __init__(self):
        pass

    def build_net(self,input_images,is_training):
        end_points = backbone(input_images,is_training)
        features = fpn(end_points,is_training)
        # self.logits = logits(features,is_training)
        # self.heat_map = heat_map(features,is_training)

        self.heat_map,self.logits = lg_hm(features,is_training)

    def loss(self,true_heat_maps,true_classify_maps):
        c_loss = classify_loss(self.logits,true_classify_maps)
        hm_loss = heat_map_loss(self.heat_map,true_heat_maps)
        total_loss = c_loss+hm_loss
        return c_loss,hm_loss,total_loss



def backbone(inputs, is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9
        , 'updates_collections': None}

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], weights_regularizer=slim.l2_regularizer(1e-4),
                        normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=[2, 2], padding='SAME'):
            end_points = {}
            conv0 = slim.conv2d(inputs, 64, scope='conv0')
            pool0 = slim.max_pool2d(conv0, kernel_size=[2, 1], stride=[2, 1], scope='pool0')

            conv1 = slim.conv2d(pool0, 64, scope='conv1')
            poo1 = slim.max_pool2d(conv1, scope='pool1')
            end_points['pool1'] = poo1

            conv2 = slim.conv2d(poo1, 128, scope='conv2')
            pool2 = slim.max_pool2d(conv2, scope='pool2')
            end_points['pool2'] = pool2

            conv3 = slim.conv2d(pool2, 256, scope='conv3')
            conv4 = slim.conv2d(conv3, 256, scope='conv4')
            pool3 = slim.max_pool2d(conv4, kernel_size=[2, 2], stride=[2, 2], scope='pool3')
            end_points['pool3'] = pool3

            conv5 = slim.conv2d(pool3, 512, scope='conv5')
            conv6 = slim.conv2d(conv5, 512, scope='conv6')
            pool4 = slim.max_pool2d(conv6, kernel_size=[2, 2], stride=[2, 2], scope='pool4')
            end_points['pool4'] = pool4

            conv7 = slim.conv2d(pool4, 512,kernel_size=[2, 2], scope='conv7')

            pool5 = slim.max_pool2d(conv7, kernel_size=[2, 2], stride=[2, 2], scope='pool4')

            end_points['pool5'] = pool5

            return end_points

def unpool(inputs,shape):
    return tf.image.resize_bilinear(inputs, size=[1,  shape])


def fpn(end_points,is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9
        , 'updates_collections': None}

    with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4),
                        normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        f = [end_points['pool5'], end_points['pool4'],
             end_points['pool3'], end_points['pool2'],
             end_points['pool1']]
        g = [None, None, None, None,None]
        h = [None, None, None, None,None]
        num_outputs = [512, 512, 256,128,64]
        h_shapes = [1,2,4,8,16,32]
        for i in range(5):
            if i == 0:
                h[i] = f[i]
            else:
                f1_1 = slim.conv2d(f[i],num_outputs[i],kernel_size=[h_shapes[i],1],padding='valid')
                c1_1 = slim.conv2d(tf.concat([g[i - 1], f1_1], axis=-1), num_outputs[i], 1)
                h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
            if i!=4:
                shape = tf.shape(f[i+1])[2]
                g[i] = unpool(h[i],shape)
        return h

def logits(features,is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9
        , 'updates_collections': None}

    with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], weights_regularizer=slim.l2_regularizer(1e-4),
                        normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        logits = slim.conv2d(features[-1],config.NUM_CLASS)
        logits =  tf.squeeze(logits, axis=1, name='logits')
        return logits

def heat_map(features,is_training):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9
        , 'updates_collections': None}

    with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], activation_fn = tf.nn.sigmoid,weights_regularizer=slim.l2_regularizer(1e-4),
                        normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        heat_map =  slim.conv2d(features[-1], 1)
        heat_map = tf.squeeze(heat_map, axis=1, name='heat_map')
        return heat_map

def lg_hm(features,is_training):
    with slim.arg_scope([slim.conv2d], kernel_size=[1, 1], activation_fn = None,weights_regularizer=slim.l2_regularizer(1e-4)):
        features =  slim.conv2d(features[-1], config.NUM_CLASS+1)
        features = tf.squeeze(features, axis=1, name='heat_map')

        heat_map = features[..., :1]
        logits = features[..., 1:]

        return heat_map, logits

def classify_loss(logits,classify_maps):
    return tf.losses.sparse_softmax_cross_entropy(labels=classify_maps,logits=logits)


def heat_map_loss(pre_heat_map,true_heat_map):
    l1_loss = tf.abs(tf.subtract(pre_heat_map, true_heat_map))
    l1_loss = tf.reduce_mean(l1_loss)
    return l1_loss


if __name__ == '__main__':
    image = tf.placeholder(dtype=tf.float32,shape=[None,64,256,3])
    end_points = backbone(image,True)
    features = fpn(end_points,True)
    logits = logits(features,True)
    print('a')

