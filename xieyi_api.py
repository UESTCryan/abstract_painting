# -*- coding: utf-8 -*-
from abart.models import create_model
from abart.options.test_options import TestOptions
from abart.data.base_dataset import get_transform
from abart.util import util
from PIL import Image
import os

def xieyi_transform(originalname,resultname):
    """
    功能:通过手绘简笔画图片，生成写意抽象画
    参数:
        originalname: 手绘简笔画路径
        resultname: 生成结果路径

    """
    # img_file = './static/image/upload/content31.jpg'
    img_file = originalname
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.model = 'test'
    opt.no_dropout = True
    opt.name = 'abart_pretrained'
    model = create_model(opt)
    model.setup(opt)
    data = Image.open(img_file).convert('RGB')
    transform = get_transform(opt, grayscale=False)
    data = transform(data)
    model.set_input(data)  # unpack data from data loader
    model.test()
    visuals = model.get_current_visuals()
    im = util.tensor2im(visuals['fake'])
    # image_dir = './static/image/upload'
    # save_path = os.path.join(image_dir, resultname)
    util.save_image(im, resultname, aspect_ratio=1.0)
    # util.save_image(im, save_path, aspect_ratio=1.0)
#
# if __name__ == '__main__':
#     xieyi_transform(resultname='target31.jpg')