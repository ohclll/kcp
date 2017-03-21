import os
import pickle
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from preprocessing import Data


def link_inference(link_in, reuse):
    T, dim = [int(i) for i in link_in.get_shape()[1:]]
    with tf.variable_scope('link', reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(link_in, name='input')
        # net = tl.layers.TimeDistributedLayer(net, tl.layers.DenseLayer,
        #                                      args={'n_units': 8, 'W_init': tf.orthogonal_initializer(),
        #                                            'name': 'dense'},name='time_dist')
        # net=tl.layers.KerasLayer(net,TimeDistributed(Dense(8,init='orthogonal')),name='keras_layer')
        net = tl.layers.LambdaLayer(net, lambda x: tf.reshape(x, [-1, dim]), name='lambda1')
        net = tl.layers.DenseLayer(net, n_units=8, W_init=tf.orthogonal_initializer(), name='share_dense')
        net = tl.layers.LambdaLayer(net, lambda x: tf.reshape(x, [-1, T, 8, 1]), name='lambda2')
        net = tl.layers.Conv2dLayer(net, shape=[3, 3, 1, 8], W_init=tf.contrib.layers.xavier_initializer(),
                                    name='conv1')
        net = tl.layers.PoolLayer(net, name='pool1')
        net = tl.layers.PReluLayer(net, a_init=tf.contrib.layers.xavier_initializer(), name='prelu1')
        net = tl.layers.Conv2dLayer(net, shape=[2, 2, 8, 16], padding='VALID',
                                    W_init=tf.contrib.layers.xavier_initializer(), name='conv2')
        net = tl.layers.PReluLayer(net, a_init=tf.constant_initializer(0.1), name='prelu2')
        net = tl.layers.FlattenLayer(net, name='flatten')
        net = tl.layers.DenseLayer(net, n_units=64, W_init=tf.contrib.layers.xavier_initializer(), name='dense1')
        net = tl.layers.PReluLayer(net, a_init=tf.constant_initializer(0.1), name='dense_prelu1')
        net = tl.layers.DenseLayer(net, n_units=6, W_init=tf.contrib.layers.xavier_initializer(), name='pred')
        return net


def mape_loss(y_true, y_pred):
    partition = tf.cast(tf.less(y_true, 1), tf.int32)
    yy_true, _ = tf.dynamic_partition(y_true, partition, 2)
    yy_pred, _ = tf.dynamic_partition(y_pred, partition, 2)
    loss = tf.reduce_mean(tf.divide(tf.abs(yy_true - yy_pred), yy_true))
    return loss


def links_to_route(link_preds, name):
    link_preds = [tl.layers.ExpandDimsLayer(l, axis=-1, name=name + 'epd{}'.format(i)) for i, l in
                  enumerate(link_preds)]
    net = tl.layers.ConcatLayer(link_preds, concat_dim=-1, name=name + 'concat')
    T, dim = [int(i) for i in net.outputs.get_shape()[1:]]
    net = tl.layers.LambdaLayer(net, lambda x: tf.reshape(x, [-1, dim]), name=name + 'lambda1')
    net = tl.layers.DenseLayer(net, n_units=1, W_init=tf.constant_initializer(1), name=name + 'share_dense')
    net = tl.layers.LambdaLayer(net, lambda x: tf.reshape(x, [-1, T]), name=name + 'lambda2')
    return net


data_dir = 'dataSets'
data = Data(data_dir)
with open(os.path.join(data_dir, 'pretrain_ftrs.pkl'), 'rb') as f:
    route_ftrs, link_ftrs, route_targets, link_targets = pickle.load(f)

link_ids = sorted(link_ftrs.keys())
num_link = len(link_ids)
link_ftr_inputs = {}
link_tgt_inputs = {}
for k in link_ids:
    if np.isnan(link_ftrs[k]).sum()>0 or np.isnan(link_targets[k]).sum()>0:
        print(k)
    link_ftr_inputs[k] = tf.constant(np.nan_to_num(np.array(link_ftrs[k], dtype='float32')))
    link_tgt_inputs[k] = tf.constant(np.nan_to_num(np.array(link_targets[k], dtype='float32')))

route_ids = sorted(route_ftrs.keys())
num_route = len(route_ids)
route_ftr_inputs = {}
route_tgt_inputs = {}
for k in route_ids:
    if np.isnan(route_ftrs[k]).sum()>0 or np.isnan(route_targets[k]).sum()>0:
        print(k)
    route_ftr_inputs[k] = tf.constant(np.nan_to_num(np.array(route_ftrs[k], dtype='float32')))
    route_tgt_inputs[k] = tf.constant(np.nan_to_num(np.array(route_targets[k], dtype='float32')))

seed = 2017
batch_size = 16
batch_link_ftr = tf.train.shuffle_batch(link_ftr_inputs, batch_size=batch_size, capacity=20,
                                        min_after_dequeue=10, enqueue_many=True, seed=seed)
batch_link_tgt = tf.train.shuffle_batch(link_tgt_inputs, batch_size=batch_size, capacity=20,
                                        min_after_dequeue=10, enqueue_many=True, seed=seed)
batch_route_ftr = tf.train.shuffle_batch(route_ftr_inputs, batch_size=batch_size, capacity=20,
                                         min_after_dequeue=10, enqueue_many=True, seed=seed)
batch_route_tgt = tf.train.shuffle_batch(route_tgt_inputs, batch_size=batch_size, capacity=20,
                                         min_after_dequeue=10, enqueue_many=True, seed=seed)
link_preds = {}
link_losses = {}
for i, k in enumerate(link_ids):
    reuse = False if i == 0 else True
    link_pred = link_inference(batch_link_ftr[k], reuse=reuse)
    link_loss = mape_loss(batch_link_tgt[k], link_pred.outputs)
    tf.losses.add_loss(link_loss)
    link_preds[k] = link_pred
    link_losses[k] = link_loss

route_preds = {}
route_losses = {}
for i, k in enumerate(route_ids):
    the_route_links = [link_preds[k] for k in data.route[k]]
    route_pred = links_to_route(the_route_links, name=k)
    route_loss = mape_loss(batch_route_tgt[k], route_pred.outputs)
    tf.losses.add_loss(route_loss)
    route_preds[k]=route_pred
    route_losses[k]=route_loss

total_loss=tf.losses.get_total_loss()
train_op=tf.train.AdamOptimizer(0.001).minimize(total_loss)
init=tf.global_variables_initializer()

loss=tf.add_n([v for k,v in route_losses.items()])/len(route_losses)

coord=tf.train.Coordinator()
with tf.Session() as sess:
    sess.run(init)
    threds=tf.train.start_queue_runners(sess,coord)
    for i in range(100000):
        _,err=sess.run([train_op,loss])
        print('\r{} loss:{:.4f}'.format(i,err),end='')
    coord.request_stop()
    coord.join(threds)




