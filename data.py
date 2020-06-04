import os
import numpy as np
import tensorflow as tf
from filecmp import dircmp
from PIL import Image
from keras.utils import to_categorical
from tqdm import trange

import params
import utils
import binvox_rw



# 用于读取路径对应的图片
def load_image(path_list):
    images = []
    for path in path_list:
        images.append(np.array(Image.open(path)))
    
    return np.array(images)


# 用于读取路径对应的体素 ([0, 1]是占用)
def load_voxel(path_list):
    with open(path_list[0], 'rb') as f: # voxel的path_list里面只有一个元素
        return np.array(to_categorical(binvox_rw.read_as_3d_array(f).data))


# 用于寻找两个路径中相同的文件路径
def find_common_paths(x_path, y_path):
    common_paths = []
    for dir_top, subdir_cmps in dircmp(x_path, y_path).subdirs.items(): # 遍历x_path和y_path中的相同目录的键值对
        for dir_bot in subdir_cmps.common_dirs: # 遍历相同目录中的相同目录
            common_paths.append(os.path.join(dir_top, dir_bot)) # 将x_path和y_path中的相同目录名和相同目录中的相同目录名连起来，加到common_paths中，也就是得到两层目录都相同的list

    return common_paths


# 用于在path中查找包含filter名的文件路径列表
def construct_path_list(path, filter):
    paths = []
    for file_name in os.listdir(path):
        if filter in file_name:
            paths.append(os.path.join(path, file_name))

    return paths


# 用于将图像和体素数据处理成npy文件
def preprocess_dataset_to_npy():
    # 获取数据路径列表
    common_paths = find_common_paths(params.image_path, params.voxel_path) # 获取图像和体素数据的相同路径，即两边都存在数据的对象
    image_paths = list(map(lambda x : os.path.join(os.getcwd(), params.image_path, x, 'rendering'), common_paths)) # 连接当前路径和共同路径，得到图像数据的绝对路径列表
    voxel_paths = list(map(lambda x : os.path.join(os.getcwd(), params.voxel_path, x), common_paths)) # 连接当前路径和共同路径，得到体素数据的绝对路径列表

    # 将路径中的数据打包存成npy
    print('正在将数据打包成npy')
    for i in trange(len(voxel_paths)):
        model_name = os.path.basename(voxel_paths[i])
        np.save('{}\\{}_x'.format(params.npy_path, model_name), load_image(construct_path_list(image_paths[i], '.png')))
        np.save('{}\\{}_y'.format(params.npy_path, model_name), load_voxel(construct_path_list(voxel_paths[i], '.binvox')))


# 用于读取npy数据路径
def load_npy_path():
    x_path_list = construct_path_list(params.npy_path, '_x.npy') # 获得npy_path中文件名包含'_x.npy'的路径列表
    y_path_list = construct_path_list(params.npy_path, '_y.npy') # 获得npy_path中文件名包含'_y.npy'的路径列表
    
    return x_path_list, y_path_list


# 用于将npy文件处理成tfrecord
def preprocess_npy_to_tfrecord():
    x_path_list, y_path_list = load_npy_path()
    
    print('正在将npy处理成tfrecord')
    with tf.io.TFRecordWriter(params.tfrecord_path) as writer:
        for i in trange(len(x_path_list)):
            x = np.load(x_path_list[i])
            y = np.load(y_path_list[i])
            x_shape = x.shape
            y_shape = y.shape
            x = x.tostring()
            y = y.tostring()
            feature = { 
                'x' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [x])),
                'x_shape' : tf.train.Feature(int64_list = tf.train.Int64List(value = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]])),
                'y' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [y])),
                'y_shape' : tf.train.Feature(int64_list = tf.train.Int64List(value = [y_shape[0], y_shape[1], y_shape[2], y_shape[3]]))
                }
            example = tf.train.Example(features = tf.train.Features(feature = feature))
            writer.write(example.SerializeToString())
        writer.close()


# 存下的tfrecord是字符串，该函数用于读取tfrecord并返回dict格式的字符串
def decode_tfrecord(input_string):
    features = {
        'x': tf.io.FixedLenFeature((), tf.string),
        'x_shape': tf.io.FixedLenFeature([4], tf.int64),
        'y': tf.io.FixedLenFeature((), tf.string),
        'y_shape': tf.io.FixedLenFeature([4], tf.int64)
    }
    feature_dict = tf.io.parse_single_example(input_string, features)

    x = feature_dict['x']
    y = feature_dict['y']
    x_shape = feature_dict['x_shape']
    y_shape = feature_dict['y_shape']

    x = tf.io.decode_raw(x, tf.uint8)
    y = tf.io.decode_raw(y, tf.float32)
    x = tf.reshape(x, x_shape)
    x = tf.map_fn(lambda _ : tf.image.resize_with_crop_or_pad(_, params.image_size, params.image_size), x) # 裁切到需要的图片大小
    y = tf.reshape(y, y_shape)
    x = tf.dtypes.cast(x, tf.float32) # 转化为float格式
    x = x / 255 # 归一化到[0,1]
    y = tf.dtypes.cast(y, tf.float32)

    return x, y


def decode_sorted_tfrecord(input_string):
    features = {
        'x': tf.io.FixedLenFeature((), tf.string),
        'x_shape': tf.io.FixedLenFeature([4], tf.int64),
        'y': tf.io.FixedLenFeature((), tf.string),
        'y_shape': tf.io.FixedLenFeature([4], tf.int64)
    }
    feature_dict = tf.io.parse_single_example(input_string, features)

    x = feature_dict['x']
    y = feature_dict['y']
    x_shape = feature_dict['x_shape']
    y_shape = feature_dict['y_shape']

    x = tf.io.decode_raw(x, tf.float32)
    y = tf.io.decode_raw(y, tf.float32)
    x = tf.reshape(x, x_shape)
    y = tf.reshape(y, y_shape)

    return x, y


# 用于读取tfrecord文件，返回tf.dataset对象
def read_tfrecord():
    dataset = tf.data.TFRecordDataset(params.tfrecord_path)
    dataset = dataset.map(decode_tfrecord)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(3)
    
    return dataset


def read_sorted_tfrecord():
    dataset = tf.data.TFRecordDataset(params.sorted_tfrecord_path)
    dataset = dataset.map(decode_sorted_tfrecord)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(3)
    
    return dataset


def sort_tfrecord():
    dataset = tf.data.TFRecordDataset(params.tfrecord_path)
    dataset = dataset.map(decode_tfrecord)
    with tf.io.TFRecordWriter(params.sorted_tfrecord_path) as writer:
        for x, y in dataset:
            x = x.numpy()
            y = y.numpy()
            x = utils.x_sort(x)
            x_shape = x.shape
            y_shape = y.shape
            x = x.tostring()
            y = y.tostring()
            feature = { 
                'x' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [x])),
                'x_shape' : tf.train.Feature(int64_list = tf.train.Int64List(value = [x_shape[0], x_shape[1], x_shape[2], x_shape[3]])),
                'y' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [y])),
                'y_shape' : tf.train.Feature(int64_list = tf.train.Int64List(value = [y_shape[0], y_shape[1], y_shape[2], y_shape[3]]))
                }
            example = tf.train.Example(features = tf.train.Features(feature = feature))
            writer.write(example.SerializeToString())
        writer.close()


