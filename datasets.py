import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom as dicom

RSZ_W = 448
RSZ_H = 448
PATCH_W = 128
PATCH_H = 128
REAL_W = 45
REAL_H = 45


def create_tf_dataset(data_path, xlsx_path, run_type, data_size, epoch, batch_size,
                      do_crop_resize=True, do_std=True, do_flip=False, do_translate=True, do_contrast=True,
                      rsz_w=RSZ_W, rsz_h=RSZ_H, translate_pixels=40, min_ratio=0.5, max_ratio=1.5):
    """
    Description
        create tf.data.Dataset from _get_maxil_data() function and pre-process data

    Param
        data_path ~ data_size : for _get_maxil_data() function
        epoch, batch_size : repetition time and parallel streaming size each
        do_resize ~ translate_pixels : for image pre-processing options

    Return
        tf.data.Dataset tensor
    """

    images, labels, infos = _get_maxil_data(data_path, xlsx_path, run_type, data_size)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels, infos))
    dataset = dataset.map(lambda x, y, z: _read_dicom_images_wrapper(x, y, z))

    if do_crop_resize:
        dataset = dataset.map(lambda x, y, z: _crop_and_resize(x, y, z, rsz_w, rsz_h))
    if do_std:
        dataset = dataset.map(lambda x, y, z: _standardization(x, y, z))
    if do_flip:
        dataset = dataset.map(lambda x, y, z: _flip(x, y, z))
    if do_translate:
        dataset = dataset.map(lambda x, y, z: _translate(x, y, z, translate_pixels))
    if do_contrast:
        dataset = dataset.map(lambda x, y, z: _random_contrast(x, y, z, min_ratio, max_ratio))

    dataset = dataset.shuffle(data_size * 2).repeat(epoch).batch(batch_size).prefetch(batch_size)

    return dataset


def _get_maxil_data(data_path, xlsx_path, run_type, data_size):
    """
    Description
        parse xlsx file into batch numpy dataset

    Param
        data_path: folder path containing raw dicom images
        xlsx_path: xlsx file path
        run_type: only available string value 'TRAIN', 'VAL', 'TEST'
        data_size: final data size

    Return
        images : batch of image path (will be used to read raw image pixels with map function)
        labels : batch of class labels and bounding box coords with [?, 10] shape
                 which are [lc, rc, lx, ly, lw, lh, rx, ry, rw, rh] through each batch
        infos : batch of information data for later use
    """

    images = []
    labels = []
    infos = []

    # A1 - Select xlsx column names by run type
    if run_type == 'TRAIN':
        lt_label, rt_label = 'LT_LABEL_USER1', 'RT_LABEL_USER1'
    elif run_type == 'VAL':
        lt_label, rt_label = 'LT_LABEL_CT', 'RT_LABEL_CT'
    elif run_type == 'TEST':
        lt_label, rt_label = 'LT_LABEL_CT', 'RT_LABEL_CT'
    else:
        print('Invalid run type')
        return

    # A2 - Read dataframe from xlsx file
    df = pd.read_excel(xlsx_path)
    df.set_index("SEED_NUM")
    df = df[['SEED_NUM', 'FILENAME', 'REVERSE', lt_label, rt_label,
             'LT_COORD_X2', 'LT_COORD_Y2', 'RT_COORD_X2', 'RT_COORD_Y2',
             'SPACING_X', 'SPACING_Y', 'ORIGINAL_H', 'ORIGINAL_W']]

    # B1 - Collect images, labels and infos
    for idx, row in df.iterrows():
        # delete invalid reverses
        if int(row['REVERSE']) >= 2:
            continue
        # delete invalid labels
        if (int(row[lt_label]) >= 4) or (int(row[rt_label]) >= 4):
            continue
        # create batch data
        else:
            lt_lbl, rt_lbl = 0, 0
            if int(row[lt_label]) > 0:
                lt_lbl = 1
            if int(row[rt_label]) > 0:
                rt_lbl = 1

            images += [os.path.join(data_path, row['FILENAME'])]
            labels += [[lt_lbl, rt_lbl,
                        int(row['LT_COORD_X2']), int(row['LT_COORD_Y2']),
                        int(REAL_W / float(row['SPACING_X'])), int(REAL_H / float(row['SPACING_Y'])),
                        int(row['RT_COORD_X2']), int(row['RT_COORD_Y2']),
                        int(REAL_W / float(row['SPACING_X'])), int(REAL_H / float(row['SPACING_Y']))]]
            infos += [[int(row['REVERSE']), int(row['ORIGINAL_W']), int(row['ORIGINAL_H'])]]

    # B2 - Set data size
    if data_size != 0:
        images = images[:data_size]
        labels = labels[:data_size]
        infos = infos[:data_size]

    # B3 - Print data size
    print('> Sinusitis data created')
    print('  Image : %d\n  Label : %d\n  Info : %d'
          % (len(images), len(labels), len(infos)))

    return images, labels, infos


def _read_dicom_images_wrapper(image, label, info):
    img = tf.py_func(_read_dicom_images, [image, info], tf.float32)
    img = tf.expand_dims(img, 2)  # add channel dimension
    return img, label, info


def _read_dicom_images(image, info):
    data = dicom.read_file(image.decode('utf-8'))
    img = data.pixel_array
    img = img.astype('float32')

    # reverse AGFA image
    if info[0] == 1:
        # max_val = np.amax(img)
        max_val = img[10][10]
        mask = np.full_like(image, max_val, img.dtype)
        img = np.subtract(mask, img)
        # img = np.subtract([4095], img)
        img = img.astype('float32')

    return img


def _crop_and_resize(image, label, info, rsz_w, rsz_h):
    # image processing
    ori_w = info[1]
    ori_h = info[2]

    # image = image[tf.to_int32((ori_h - ori_w) / 2):tf.to_int32((ori_h + ori_w) / 2), :, :]
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize_bilinear(image, [rsz_h, rsz_w])  # this function requires rank 4 (need to expand & squeeze)
    image = tf.squeeze(image, axis=0)
    image = tf.cast(image, tf.float32)

    # label processing
    h_ratio = tf.divide(rsz_h, ori_h)
    w_ratio = tf.divide(rsz_w, ori_w)
    adjust_coord = [0, 0, 0, (ori_h - ori_w) / 2, 0, 0, 0, (ori_h - ori_w) / 2, 0, 0]
    adjust_scale = [1., 1., w_ratio, h_ratio, w_ratio, h_ratio, w_ratio, h_ratio, w_ratio, h_ratio]

    # label = tf.subtract(label, adjust_coord)
    label = tf.to_float(label)
    label = tf.to_int32(tf.multiply(label, adjust_scale))

    return image, label, info


def _standardization(image, label, info):
    image = tf.image.per_image_standardization(image)

    return image, label, info


def _flip(image, label, info):
    image = tf.image.random_flip_left_right(image)

    return image, label, info


def _translate(image, label, info, pixel=10):
    tf.random.set_random_seed(1234)
    rand = tf.random_uniform([2], -1 * pixel, pixel, dtype=tf.float32)
    image = tf.contrib.image.translate(image, rand)
    adjust_coord = [0, 0, rand[0], rand[1], 0, 0, rand[0], rand[1], 0, 0]
    label += adjust_coord

    return image, label, info


def _random_contrast(image, label, info, min_ratio, max_ratio):
    image = tf.image.random_contrast(image, min_ratio, max_ratio)

    return image, label, info
