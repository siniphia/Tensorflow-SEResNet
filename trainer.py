import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import models
import metrics
import datasets as ds
import visualizer as vis

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# path params
PROJECT_NAME = 'MULTISINUS'
PROJECT_PATH = "/home/lkr/PROJECTS/multisinus_torch/"
DATA_PATH = "/data/SNUBH/sinusitis/RAW_DEID/"
TRAIN_PATH = "sinus_train_deid/"
VTSNUBH_PATH = "sinus_vtsnubh_deid/"
VTSNUH_PATH = "sinus_vtsnuh_deid/"
XLSX_PATH = "/home/lkr/DATASETS/sinusitis/"

# model params
REG_WIDTH = ds.RSZ_W
REG_HEIGHT = ds.RSZ_H
REG_CHANNEL = 1
CLS_WIDTH = ds.PATCH_W
CLS_HEIGHT = ds.PATCH_H
CLS_CHANNEL = 1
CLASSES = 2  # sinus num to diagnose

# training params
IS_TRAINING = True
TRIAL_NAME = "trial12/"
LEARNING_RATE = 0.001
BATCH_SIZE = 20
EPOCH = 200

# saver, tensorboard, heatmap parameters
SAVE_CKPT = True
SAVE_SMR = False
SAVE_CAM = False
SAVE_PLOT = False

LOAD_CKPT = False
LOAD_TRIAL = "trial12/"  # trial to load ckpt
LOAD_FILE = "13500step"  # steps to load ckpt


# Create folders for current trial
def create_workspace(project_path=PROJECT_PATH, result_path='result/', trial_path=TRIAL_NAME,
                     ckpt_path='checkpoint/', smr_path='tensorboard/', cam_path='cam/'):
    if not os.path.exists(project_path + result_path + trial_path):
        os.makedirs(project_path + result_path + trial_path)
        print('> Created workspace for %s' % trial_path)

    if not os.path.exists(project_path + result_path + trial_path + ckpt_path):
        os.makedirs(project_path + result_path + trial_path + ckpt_path)
        print('> Created checkpoint folder for %s' % trial_path)

    if not os.path.exists(project_path + result_path + trial_path + smr_path):
        os.makedirs(project_path + result_path + trial_path + smr_path)
        print('> Created Tensorboard folder for %s' % trial_path)

    if not os.path.exists(project_path + result_path + trial_path + cam_path):
        os.makedirs(project_path + result_path + trial_path + cam_path)
        print('> Created CAM folder for %s' % trial_path)


def train(train_dataset, val_dataset,
          save_ckpt=SAVE_CKPT, load_ckpt=LOAD_CKPT, save_smr=SAVE_SMR, save_cam=SAVE_CAM,
          project_path=PROJECT_PATH, result_path='result/', trial_path=TRIAL_NAME, ckpt_path='checkpoint/',
          smr_path='tensorboard/', cam_path='cam/', load_trial=LOAD_TRIAL, load_file=LOAD_FILE):

    with tf.Graph().as_default():
        # A1 - Dataset Graphs
        train_iterator = train_dataset.make_one_shot_iterator()
        train_next_element = train_iterator.get_next()
        val_iterator = val_dataset.make_one_shot_iterator()
        val_next_element = val_iterator.get_next()
        print('> Dataset Loaded')

        # A2 - Model Graphs
        img_ph = tf.placeholder(tf.float32, shape=[None, REG_HEIGHT, REG_WIDTH, REG_CHANNEL])
        lbl_ph = tf.placeholder(tf.float32, shape=[None, CLASSES * (1 + 4)])

        reg_model = models.SeResNetDouble4_448('regressor', img_ph, REG_CHANNEL, CLASSES * 4)
        bridge_model = models.PatchProcessor('bridge', img_ph, lbl_ph[:, 2:], batch_size=BATCH_SIZE)
        bridge_model.random_distortion(BATCH_SIZE * 2, do_translate=True, do_contrast=False, do_flip=False)
        cls_model = models.SeResNetDouble3_128('classifier', bridge_model.cropped_imgs, CLS_CHANNEL, CLASSES)
        print('> Model Loaded')

        # A3 - Metric Graphs
        step = tf.train.create_global_step()
        reg_logits = reg_model.logits
        cls_logits = cls_model.logits

        reg_loss = metrics.get_regression_loss(reg_logits, lbl_ph[:, 2:], method='L2')
        cls_loss = metrics.get_logistic_loss(cls_logits, lbl_ph[:, :2])
        total_loss = reg_loss + cls_loss

        pred = metrics.get_pred(cls_logits)
        acc = metrics.get_accuracy(pred, lbl_ph[:, :2])
        optimizer = metrics.get_optimizer(learning_rate=LEARNING_RATE, loss=total_loss, global_step=step)

        # A4 - Misc
        saver = tf.train.Saver(max_to_keep=30)
        root_path = os.path.join(project_path, result_path, trial_path)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            if load_ckpt:
                saver.restore(sess, os.path.join(project_path, result_path, load_trial, ckpt_path, load_file))
                print('> Model weights restored (%s from %s)' % (load_file, load_trial))

            print('> Start training')
            print('\tbatch size : ', BATCH_SIZE)
            print('\tlearning rate : ', LEARNING_RATE)
            print('\tepoch : ', EPOCH)

            while True:
                try:
                    # (1) Training
                    step_tensor = tf.train.get_global_step()
                    step = tf.train.global_step(sess, step_tensor)

                    train_image, train_label, _ = sess.run(train_next_element)
                    train_reg_logits, train_reg_loss, train_cls_loss, train_patch, _\
                        = sess.run([reg_logits, reg_loss, cls_loss, bridge_model.cropped_imgs, optimizer],
                                   feed_dict={img_ph: train_image, lbl_ph: train_label})

                    if step > 15000:
                        break

                    # print('reg logit : ', train_reg_logits)
                    # if step % 10 == 0:
                    #     plt.imshow(np.squeeze(np.asarray(train_patch[0])))
                    #     plt.show()

                    # ckpt
                    if save_ckpt and step % 300 == 0 and step > 0:
                        saver.save(sess, os.path.join(root_path, ckpt_path, str(step) + 'step'))
                        print('> Model saved')

                    # (2) Validation
                    if step % 100 == 0:
                        val_image, val_label, _ = sess.run(val_next_element)
                        val_acc, val_pred, val_reg_logits, val_reg_loss, val_cls_loss\
                            = sess.run([acc, pred, tf.to_int32(reg_logits), reg_loss, cls_loss],
                                       feed_dict={img_ph: val_image, lbl_ph: val_label})

                        print('[%.4d] acc : %f' % (step, val_acc))
                        tempimg = np.asarray(val_image[0])
                        tempimg = np.squeeze(tempimg, axis=2)
                        templbl = np.asarray(val_reg_logits[0])
                        templt = np.asarray(val_pred[0])
                        templtlbl = np.asarray(val_label[0][0])
                        temprtlbl = np.asarray(val_label[0][1])
                        vis.draw_bounding_box(tempimg, templbl[0], templbl[1], templbl[2], templbl[3], templbl[4],
                                              templbl[5], templbl[6], templbl[7], templt, templt, templtlbl, temprtlbl)
                        print('> Regressor : Train Loss - %0.3f, Val Loss - %0.3f\r' % (train_reg_loss, val_reg_loss))
                        print('> Classifier : Train Loss - %0.3f, Val Loss - %0.3f\r' % (train_cls_loss, val_cls_loss))

                except tf.errors.OutOfRangeError:
                    print('> Done training')
                    break


def test(test_dataset,
         save_ckpt=SAVE_CKPT, load_ckpt=LOAD_CKPT, save_smr=SAVE_SMR, save_cam=SAVE_CAM,
         project_path=PROJECT_PATH, result_path='result/', trial_path=TRIAL_NAME, ckpt_path='checkpoint/',
         smr_path='tensorboard/', cam_path='cam/', load_trial=LOAD_TRIAL, load_file=LOAD_FILE):

    with tf.Graph().as_default():
        # A1 - Dataset Graphs
        test_iterator = test_dataset.make_one_shot_iterator()
        test_next_element = test_iterator.get_next()
        print('> Dataset Loaded')

        # A2 - Model Graphs
        img_ph = tf.placeholder(tf.float32, shape=[None, REG_HEIGHT, REG_WIDTH, REG_CHANNEL])
        lbl_ph = tf.placeholder(tf.float32, shape=[None, CLASSES * (1 + 4)])
        reg_model = models.SeResNetDouble4_448('regressor', img_ph, REG_CHANNEL, CLASSES * 4)
        bridge_model = models.PatchProcessor('bridge', img_ph, lbl_ph[:, 2:], batch_size=BATCH_SIZE)
        # bridge_model = models.PatchProcessor('bridge', img_ph, reg_model.logits, batch_size=BATCH_SIZE)
        cls_model = models.SeResNetDouble3_128('classifier', bridge_model.cropped_imgs, CLS_CHANNEL, CLASSES)
        print('> Model Loaded')

        # A3 - Metric Graphs
        step = tf.train.create_global_step()
        reg_logits = reg_model.logits
        cls_logits = cls_model.logits

        reg_loss = metrics.get_regression_loss(reg_logits, lbl_ph[:, 2:], method='L1')
        cls_loss = metrics.get_logistic_loss(cls_logits, lbl_ph[:, :2])

        pred = metrics.get_pred(cls_logits)
        prob = metrics.get_prob(cls_logits)
        pos_prob = metrics.get_pos_prob(cls_logits)
        acc = metrics.get_accuracy(pred, lbl_ph[:, :2])
        auc = metrics.get_auc_tf(lbl_ph[:, :2], pos_prob)
        iou = metrics.get_iou(reg_logits, lbl_ph[:, 2:])
        reg_cam = reg_model.get_cam(128, 128)

        # A4 - Misc
        saver = tf.train.Saver()
        sum_iou = 0
        pos = 0  # correct inferences
        neg = 0  # false inferences

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if load_ckpt:
                saver.restore(sess, os.path.join(project_path, result_path, load_trial, ckpt_path, load_file))
                print('> Model weights restored (%s from %s)' % (load_file, load_trial))

            print('> Start testing')

            i = 0
            while True:
                try:
                    # (1) Testing
                    step_tensor = tf.train.get_global_step()
                    step = tf.train.global_step(sess, step_tensor)
                    test_image, test_label, _ = sess.run(test_next_element)
                    test_reg_logits, test_cls_logits, test_reg_loss, test_cls_loss, test_pred, test_pos_prob, test_auc, test_iou, test_cam\
                        = sess.run([reg_logits, cls_logits, reg_loss, cls_loss, pred, pos_prob, auc, iou, reg_cam],
                                   feed_dict={img_ph: test_image, lbl_ph: test_label})

                    # np results
                    test_cls_label = np.concatenate([test_label[:, 0:1], test_label[:, 1:2]], axis=0)

                    # iou results
                    if i == 0:
                        sum_iou = test_iou
                    else:
                        sum_iou = np.concatenate([sum_iou, test_iou], axis=0)

                    # plt.imshow(np.squeeze(np.asarray(test_image[0]), axis=2))
                    # plt.imshow(np.squeeze(np.asarray(test_cam[0]), axis=2))
                    # plt.show()

                    # numpy array for cls auc
                    if i == 0:
                        label_list = test_cls_label
                        prob_list = test_pos_prob
                    else:
                        label_list = np.concatenate([label_list, test_cls_label], axis=0)
                        prob_list = np.concatenate([prob_list, test_pos_prob], axis=0)

                    # cls accuracy
                    for j in range(len(test_cls_label)):
                        if int(test_cls_label[j]) == int(test_pred[j]):
                            pos += 1
                        else:
                            neg += 1

                    # plot bounding box
                    if SAVE_PLOT:
                        for k in range(BATCH_SIZE):
                            image = np.squeeze(np.asarray(test_image[k]), axis=2)
                            label = np.asarray(test_label[k])
                            reg_pred = np.asarray(test_reg_logits[k])
                            cls_lt_pred = np.asarray(test_pred[k])
                            cls_rt_pred = np.asarray(test_pred[k + BATCH_SIZE])
                            vis.plot_bounding_box(image, label, reg_pred, cls_lt_pred, cls_rt_pred, BATCH_SIZE * i + (k + 1),
                                                  os.path.join(project_path, result_path, load_trial), LOAD_FILE, True)

                    i += 1

                except tf.errors.OutOfRangeError:
                    acc = pos / (pos + neg)
                    fpr, tpr, cutoff, auc_val = metrics.get_roc(label_list, prob_list)
                    # total_iou = sum_iou / i
                    print('TF AUC : ', test_auc)
                    print('Correct : %d, Incorrect : %d, Accuracy : %0.3f' % (pos, neg, acc))
                    print('FPR : %f, TPR : %f, Cutoff : %f, AUC : %0.3f' % (fpr, tpr, cutoff, auc_val))
                    # print(np.squeeze(sum_iou))

                    t_50, f_50 = 0, 0
                    t_75, f_75 = 0, 0
                    for i in range(len(sum_iou)):
                        if sum_iou[i] >= 0.5:
                            t_50 += 1
                        else:
                            f_50 += 1

                    for i in range(len(sum_iou)):
                        if sum_iou[i] >= 0.75:
                            t_75 += 1
                        else:
                            f_75 += 1

                    print('mAP 0.50 : ', t_50 / (t_50 + f_50))
                    print('mAP 0.75 : ', t_75 / (t_75 + f_75))

                    break


if __name__ == '__main__':
    create_workspace()
    if IS_TRAINING:
        train_dataset = ds.create_tf_dataset(DATA_PATH + 'sinus_train_deid/', XLSX_PATH + '01_train.xlsx',
                                             'TRAIN', 8000, EPOCH, BATCH_SIZE, do_translate=False)
        print('> Training Dataset Ready')

        val_dataset = ds.create_tf_dataset(DATA_PATH + 'sinus_vtsnubh_deid/', XLSX_PATH + '02_val_snubh.xlsx',
                                           'VAL', 100, EPOCH, BATCH_SIZE)

        print('> Validation Dataset Ready')
        train(train_dataset, val_dataset)
    else:
        snubh_dataset = ds.create_tf_dataset(DATA_PATH + 'sinus_vtsnubh_deid/', XLSX_PATH + '03_test_snubh.xlsx',
                                             'VAL', 140, 1, BATCH_SIZE, do_contrast=False, do_translate=False)
        test(snubh_dataset)
        print('> SNUBH Test Finished')

        snuh_dataset = ds.create_tf_dataset(DATA_PATH + 'sinus_vtsnuh_deid/', XLSX_PATH + '04_test_snuh.xlsx',
                                            'VAL', 160, 1, BATCH_SIZE, do_contrast=False, do_translate=False)
        test(snuh_dataset)
        print('> SNUH Test Finished')

    print('> Exit program')
