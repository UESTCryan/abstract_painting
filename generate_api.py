#!/user/bin/python3
# -* utf-8 *-
# © Jian Wang
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import operator
import utils
from PIL import Image
import style
import logging

distribution = utils.Distribution()
logging.basicConfig(level=logging.INFO)


def blocked(file):
    """
    功能: 读取图片，然后聚类和提取色块,计算每种颜色占比
    参数:
        file: 图片文件
    返回:
        提取的色块 [[ 颜色[r, g, b], 大小, 占图像整体比例],  ...]
        每种色彩占比 [[ 颜色[r, g, b], 占图像整体比例],  ...]
    """
    im = Image.open(file)
    logging.info("Processing: {0}".format(file))
    im = im.resize((128, 128))
    width, height = im.size
    im = np.reshape(np.array(im), [-1, 3])
    k = utils.find_k(im)
    # 聚类
    # 返回： 每个像素点所属种类；中心点的值
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(im)
    lbs = kmeans.labels_
    centers = kmeans.cluster_centers_

    for i in range(k):
        im[lbs[:] == i] = centers[i]
    im = np.reshape(im, [height, width, 3])
    # # show the result
    # plt.imshow(im)
    # plt.show()

    # 提取色块 格式:[[ 颜色[r, g, b], 大小, 占图像整体比例],  ...]
    blocks = utils.get_color_block(height, width, im)

    # 计算每种颜色的占比（用不着）
    pixes_num = height * width
    cate_num_list = np.zeros([k, ])
    for j in range(k):
        cate_num_list[j] = 0
        for lb in lbs:
            if lb == j:
                cate_num_list[j] += 1
    percentage_num_list = cate_num_list / pixes_num

    # # show percentage by bar graph
    # bar_width = 0.35
    # opacity = 1
    # index = np.arange(k)
    # rects = plt.bar(index, percentage_num_list, bar_width, alpha=opacity, color=centers / 255)
    # plt.ylim(0, 1)
    # plt.xlabel('Color')
    # plt.ylabel('Percentage')
    # plt.tight_layout()
    # plt.show()

    percentage_list = [[np.array(centers[k], np.uint8).tolist(), p]
                       for k, p in enumerate(percentage_num_list)]

    return blocks, percentage_list


def plot(blocks, percentage_list, out_path, refer_img, height=2000, width=2000):
    """
    功能: 重构图像（绘制抽象画）
    参数:
        blocks: 从图片中提取的色块 [[ 颜色[r, g, b], 大小, 占图像整体比例],  ...]
        percentage_list: 图片每种色彩占比[[ 颜色[r, g, b], 占图像整体比例],  ...]
        out_path: 重构图像路径
        refer_img: 参考分布图像的路径
        height: 重构图像高度
        width: 重构图像宽度
    """
    # 新建画布
    img = np.zeros([height, width, 3], dtype=np.uint8)

    # ---------------------------------------------
    # 画背景
    # 将提取的最大颜色占比画成背景
    block_lens = [x[1] for x in percentage_list]
    bgcolor = percentage_list[block_lens.index(max(block_lens))][0]

    img[:, :] = bgcolor
    # 画完背景后去掉
    blocks = [b for b in blocks if not operator.eq(b[0], bgcolor)]

    # ---------------------------------------------
    # 分布参考点
    loc_distribution = distribution(refer_img)
    block_num = len(blocks)
    # 补充分布参考点
    if block_num > len(loc_distribution):
        num = len(loc_distribution) // block_num
        for _ in range(num):
            loc_distribution.extend(loc_distribution)
    # ---------------------------------------------
    # 开始画图
    img = style.style_1(img, blocks, block_num, width, height, loc_distribution)
    # plt.imshow(img)
    # plt.show()

    # ---------------------------------------------
    # 保存图片
    plt.imsave(out_path, img)

    # cv.namedWindow("Image")
    # cv.imshow("Image", img)
    # cv.waitKey(0)
    # # 释放窗口
    # cv.destroyAllWindows()

