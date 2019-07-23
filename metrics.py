import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def get_logistic_loss(logits, labels, one_hot=False):
    # cls logits : (64, 2)
    # cls labels : (32, 2)
    lt_maxil_lbl = labels[:, 0:1]  # (32, 1)
    rt_maxil_lbl = labels[:, 1:2]  # (32, 1)
    labels = tf.concat([lt_maxil_lbl, rt_maxil_lbl], axis=0)  # (64, 1)
    labels = tf.to_int32(labels)
    labels = tf.squeeze(tf.one_hot(labels, depth=2))  # (64, 2)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return loss


def get_regression_loss(logits, labels, method):
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)

    if method == 'L1':
        loss = tf.losses.huber_loss(labels, logits)
    elif method == 'L2':
        loss = tf.nn.l2_loss((logits - labels) / 1024)
    else:
        loss = None
        print('Invalid loss function type')

    return loss


def get_prob(logits):
    # eg. logits : (64, 2)
    softmax = tf.nn.softmax(logits)  # (64, 2)
    prob = tf.reduce_max(softmax, axis=1)  # (64, )
    prob = tf.expand_dims(prob, axis=1)  # (64, 1)

    return prob


# probabilities for true label only
def get_pos_prob(logits):
    softmax = tf.nn.softmax(logits)
    pos_prob = tf.expand_dims(softmax[:, 1], axis=1)

    return pos_prob


def get_pred(logits):
    # eg. logits : (64, 2)
    softmax = tf.nn.softmax(logits)  # (64, 2)
    pred = tf.argmax(softmax, axis=1)  # (64, )
    pred = tf.expand_dims(pred, axis=1)  # (64, 1)

    return pred


def get_accuracy(pred, labels):
    pred = tf.cast(pred, tf.int32)

    lt_maxil_lbl = labels[:, 0:1]
    rt_maxil_lbl = labels[:, 1:2]
    labels = tf.concat([lt_maxil_lbl, rt_maxil_lbl], axis=0)
    labels = tf.cast(labels, tf.int32)

    acc = tf.reduce_mean(tf.cast(tf.equal(pred, labels), dtype=tf.float32))

    return acc


def get_optimizer(loss, learning_rate, global_step):
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.95)\
        .minimize(loss, global_step=global_step)

    return opt


# for (x, y, w, h) style
def get_iou(bb1, bb2, epsilon=0.0001):

    stacked_bb1 = tf.concat([bb1[:, 0:4], bb1[:, 4:8]], axis=0)
    stacked_bb2 = tf.concat([bb2[:, 0:4], bb2[:, 4:8]], axis=0)

    x12, y12, w1, h1 = tf.split(stacked_bb1, 4, axis=1)
    x22, y22, w2, h2 = tf.split(stacked_bb2, 4, axis=1)

    x11 = x12 - w1 + 1
    y11 = y12 - h1 + 1
    x21 = x22 - w2 + 1
    y21 = y22 - h2 + 1

    xi1 = tf.maximum(x11, x21)
    yi1 = tf.maximum(y11, y21)
    xi2 = tf.minimum(x12, x22)
    yi2 = tf.minimum(y12, y22)

    inter = tf.maximum(xi2 - xi1 + 1, 0) * tf.maximum(yi2 - yi1 + 1, 0)
    union = (w1 * h1 + w2 * h2) - inter

    return inter / (union + epsilon)


def get_auc_tf(labels, pred):
    lt_maxil_label = labels[:, 0:1]
    rt_maxil_label = labels[:, 1:2]
    label = tf.concat([lt_maxil_label, rt_maxil_label], axis=0)
    auc_val = tf.metrics.auc(label, pred)

    return auc_val


def get_roc(label, prob):
    fpr, tpr, thresholds = roc_curve(label, prob, pos_label=1)
    auc_val = np.round_(auc(fpr, tpr), 2)
    tmp_cutoff = tpr[0] - fpr[0]
    opt_cutoff = 0.5
    opt_index = 0

    for i in range(len(fpr)):
        if tpr[i] - fpr[i] > tmp_cutoff:
            tmp_cutoff = tpr[i] - fpr[i]
            opt_cutoff = thresholds[i]
            opt_index = i

    plt.plot(fpr, tpr, 'b', label='AUC = %.2f' % auc_val)
    plt.title('Sinusitis Diagnosis')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

    return fpr[opt_index], tpr[opt_index], opt_cutoff, auc_val
