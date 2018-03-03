import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from tqdm import trange
import os
import time

flags = tf.flags

flags.DEFINE_integer("evaluate_every", 100, "Number of steps before evaluation")

flags.DEFINE_integer("batch_size", 64, "Size of batch")
flags.DEFINE_integer("width", 28, "Input width")
flags.DEFINE_integer("height", 28, "Input height")
flags.DEFINE_integer("num_classes", 10, "Number of classes")

flags.DEFINE_float("learning_rate", 1e-3, "Learning rate parameter")
flags.DEFINE_float("weight_decay", 0.01, "Lambda weight decay, if None no loss will be applied")
flags.DEFINE_float("dropout", .5, "Lambda weight decay, if None no loss will be applied")
flags.DEFINE_integer("epochs", 25, "Number of epochs")

flags.DEFINE_boolean("onehot", False, "Onehot encoding or sparse")
flags.DEFINE_string("log_dir", "train_dir", "Folder where to save logs")

FLAGS = flags.FLAGS


def l1_loss(params):
    return tf.reduce_sum(tf.abs(params))


def create_variable(name, shape, weight_decay=None, loss=tf.nn.l2_loss):
    with tf.device("/cpu:0"):
        var = tf.get_variable(name, dtype=tf.float32, shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.05))

    if weight_decay:
        wd = loss(var) * weight_decay
        tf.add_to_collection("weight_decay", wd)

    return var


def hidden_layer(input, weights, bias, name, dropout_prob=1., activation=tf.nn.relu):
    with tf.name_scope(name):
        out = tf.matmul(input, weights) + bias
        out = activation(out)
        out = tf.nn.dropout(out, dropout_prob)

        return out


def outpu_layer(input, weights, bias, name, dropout=1., activation=tf.nn.relu):
    return hidden_layer(input, weights, bias, name, dropout, activation)


def compute_loss(name_scope, logits, labels, sparse=True):
    if not sparse:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    cross_entropy_mean = tf.reduce_mean(
        cross_entropy
    )

    tf.summary.scalar(
        name_scope + '_cross_entropy',
        cross_entropy_mean
    )

    weight_decay_loss = tf.get_collection('weight_decay')

    if len(weight_decay_loss) > 0:
        tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss))
        tf.summary.histogram(name_scope + '_weight_decay_loss', weight_decay_loss)

        # Calculate the total loss for the current tower.
        total_loss = cross_entropy_mean + weight_decay_loss
        tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss))
    else:
        total_loss = cross_entropy_mean

    return total_loss


def compute_accuracy(logits, labels, sparse=True):
    if not sparse:
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    else:
        correct_pred = tf.equal(tf.argmax(logits, 1), labels)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def train():
    mnist = input_data.read_data_sets('../datasets/MNIST_data/', one_hot=FLAGS.onehot)
    input_size = FLAGS.height * FLAGS.width

    train_x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, input_size], name="input_placeholder")
    train_y = tf.placeholder(tf.int64, shape=[FLAGS.batch_size], name="labels_placeholder")
    dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout_placeholder")

    weights = {
        'w1': create_variable("w1", [input_size, 1024], FLAGS.weight_decay),
        'w2': create_variable("w2", [1024, 512], FLAGS.weight_decay),
        'w3': create_variable("w3", [512, 64], FLAGS.weight_decay),
        'wout': create_variable("wout", [64, FLAGS.num_classes], FLAGS.weight_decay)
    }

    bias = {
        'b1': create_variable("b1", [1024]),
        'b2': create_variable("b2", [512]),
        'b3': create_variable("b3", [64]),
        'bout': create_variable("bout", [FLAGS.num_classes])
    }

    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.9)

    net = hidden_layer(train_x, weights=weights['w1'], bias=bias["b1"], name="hidden_1")
    net = hidden_layer(net, weights=weights['w2'], bias=bias["b2"], name="hidden_2")
    net = hidden_layer(net, weights=weights['w3'], bias=bias["b3"], name="hidden_3",
                       dropout_prob=dropout_placeholder)

    with tf.name_scope("Output"):
        net = tf.reshape(net, [FLAGS.batch_size, -1])
        logits = outpu_layer(net, weights=weights['wout'],
                             bias=bias["bout"], name="output_layer", activation=tf.identity)

    with tf.name_scope("Loss"):
        loss = compute_loss("reg_example", logits=logits, labels=train_y)

    with tf.name_scope("Accuracy"):
        accuracy = compute_accuracy(logits=logits, labels=train_y, sparse=not FLAGS.onehot)

    train_op = optimizer.minimize(loss)
    merged = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        current_exec = str(time.time())

        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, current_exec, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, current_exec, "val"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, current_exec, "test"), sess.graph)

        print("Start training")

        max_steps = int(mnist.train.num_examples / FLAGS.batch_size)
        print("Epoch every {} steps".format(max_steps))

        max_steps = max_steps * FLAGS.epochs
        total_steps = trange(max_steps)

        t_loss, v_loss, t_acc, v_acc = .0, .0, .0, .0
        for step in total_steps:
            train_samples, train_labels = mnist.train.next_batch(FLAGS.batch_size)
            _, t_loss = sess.run([train_op, loss], feed_dict={
                train_x: train_samples,
                train_y: train_labels,
                dropout_placeholder: FLAGS.dropout
            })

            t_loss = np.mean(t_loss)
            total_steps.set_description("Loss: {:.4f}/{:.4f} - Accuracy: {:.3f}/{:.3f}"
                                        .format(t_loss, v_loss, t_acc, v_acc))

            if step % FLAGS.evaluate_every == 0 or (step + 1) == max_steps:
                summary, t_acc = sess.run([merged, accuracy], feed_dict={
                    train_x: train_samples,
                    train_y: train_labels,
                    dropout_placeholder: 1.
                })
                train_writer.add_summary(summary, global_step=step)

                val_samples, val_labels = mnist.validation.next_batch(FLAGS.batch_size)
                summary, v_loss, v_acc = sess.run([merged, loss, accuracy], feed_dict={
                    train_x: val_samples,
                    train_y: val_labels,
                    dropout_placeholder: 1.
                })
                v_loss = np.mean(v_loss)
                val_writer.add_summary(summary, global_step=step)

                total_steps.set_description("Loss: {:.4f}/{:.4f} - Accuracy: {:.3f}/{:.3f}"
                                            .format(t_loss, v_loss, t_acc, v_acc))

        print("Training done")

        print("Start testing")
        max_steps = int(mnist.test.num_examples / FLAGS.batch_size)
        total_steps = trange(max_steps)

        mean_acc = []
        for step in total_steps:
            test_samples, test_labels = mnist.test.next_batch(FLAGS.batch_size)
            summary, t_loss, t_acc = sess.run([merged, loss, accuracy], feed_dict={
                train_x: test_samples,
                train_y: test_labels,
                dropout_placeholder: 1.
            })

            mean_acc.append(t_acc)
            t_loss = np.mean(t_loss)
            test_writer.add_summary(summary, global_step=step)
            total_steps.set_description("Loss: {:.4f} - Accuracy: {:.3f}"
                                        .format(t_loss, t_acc))
        print("Testing is done")
        print("Test mean accuracy", np.mean(mean_acc))


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
