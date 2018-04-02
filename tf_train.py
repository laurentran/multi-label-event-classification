from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    # input layer
    input_layer = tf.reshape(features["x"], [-1, 8, 8, 1])

    # conv layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=32, 
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    # pool layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[2,2],
        strides=2)

    # conv layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu)

    # pool layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[2,2],
        strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 2 * 2 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.3,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(
        inputs=dropout,
        units=8)
    
    predictions = {
        # predictions for PREDICT and EVAL
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss for TRAIN and EVAL
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # training operation for TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval metrics for EVAL
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # load data
    df = pd.read_csv('data/SampleData.csv')
    X = df.drop(['Label'], axis=1).as_matrix()
    #X = tf.cast(X, tf.float32)
    Y = df['Label']
    Y = Y.as_matrix()
    train_data, eval_data, train_labels, eval_labels = train_test_split(X, Y, test_size=0.3, stratify=Y)

    # estimator
    clf = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp")

    # logging
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # train model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    clf.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # evaluate model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = clf.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()