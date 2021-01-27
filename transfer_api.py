# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import os
import net
from PIL import Image
import numpy as np

models_dict = {'Cubist': 'cubist.ckpt-done',
               'Denoised_Starry': 'denoised_starry.ckpt-done',
               'Feathers': 'feathers.ckpt-done',
               'Scream': 'scream.ckpt-done',
               'Udnie': 'udnie.ckpt-done',
               'Wave': 'wave.ckpt-done',
               'Painting': 'painting.ckpt-2000',

               }


def style_transform(img_file, result_file, style):
    """
    功能: 迁移响应油画风格
    参数:
        img_file: 上传的图片
        result_file: 返回生成结果图片的路径
        style: 迁移的风格，modes_dict里，如：Cubist
    """
    model_file = 'model/' + models_dict[style]
    with tf.Graph().as_default():
        with tf.compat.v1.Session().as_default() as sess:
            # png = img_file.lower().endswith('png')
            # img_bytes = tf.compat.v1.read_file(img_file)
            # image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
            image = Image.open(img_file)
            im_np=np.asarray(image)
            im_tf=tf.convert_to_tensor(im_np)
            image = tf.expand_dims(im_tf, 0)
            print(image)
            generated = net.transform_network(image, training=False)
            generated = tf.squeeze(generated, [0])
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
            saver.restore(sess, model_file)

            start_time = time.time()
            generated = sess.run(generated)
            generated = tf.cast(generated, tf.uint8)
            end_time = time.time()
            print('Elapsed time: %fs' % (end_time - start_time))
            generated_file = result_file
            with open(generated_file, 'wb') as img:
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                return generated_file
# style_transform('test0.jpg','d.jpg','Cubist')
