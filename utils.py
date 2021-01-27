#!/user/bin/python3
# -* utf-8 *-
# © Jian Wang
from scipy import spatial
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import random
import cv2 as cv


def stay_away_from_edge(height, width, x, y, r, margin_prob=0.2):
    """
    功能: 让图形远离边界处，只要图形与边界接触，将其拉回来
    参数:
        height: 画布的高度
        width: 画布的宽度
        x: 当前图形中心点x坐标
        y: 当前图形中心点y坐标
        r: 图形的半径
        margin_prob: 远离边界的距离（百分比计算）
    返回: 调整过后的坐标(int x, int y)
    """
    x_, y_ = x, y
    if (x + r) > width:
        x_ = width - r - (width * margin_prob)
    if (x - r) < 0:
        x_ = r + (width * margin_prob)
    if (y + r) > height:
        y_ = height - r - (height * margin_prob)
    if (y - r) < 0:
        y_ = r + (height * margin_prob)
    return int(x_), int(y_)


def find_k(im):
    """
    功能: 找到最佳的聚类K值，K值不能小于5，依此往上找最佳的K值，
        最佳的K值应该再保证色彩向量夹角不小threshold的情况尽可能大，
        但K值不能超过20
    参数:
        im: 图片 Numpy ndarray格式
    返回: 最佳的K值(int)
    """
    best_k = 5
    threshold = 4   # 2.6
    head, rear = 5, 20
    zoom = 10000
    for k in range(head, rear):
        kmeans = KMeans(n_clusters=k, max_iter=100).fit(im)
        centers = kmeans.cluster_centers_
        # calcu minimum distance among centers
        dist_list = []
        for l in range(k):
            for m in range(l + 1, k):
                # calcu Euclidean Distance
                dist = spatial.distance.cosine(centers[l], centers[m])
                dist_list.append(dist)
        mini_dist = min(dist_list) * zoom
        # print("minimum distance {0}".format(mini_dist))
        if mini_dist <= threshold:
            best_k = k-1
            break
    return best_k


def get_color_block(height, width, im):
    """
    功能: 在图像色彩聚类后提取图像中每一个色块，使用广度优先算法
    参数:
        height: 图像的高度
        width: 图像的宽度
        im: 图像 Numpy ndarray 格式
    返回: 提取的色块 [[ 颜色[r, g, b], 大小, 占图像整体比例],  ...]
    """
    block, locs = [], []
    for i in range(height):
        for j in range(width):
            locs.append((i, j))
    queue, visited = [], []

    def neighbors(loc):
        nbs = []
        if loc[0] + 1 < height:
            if (im[(loc[0] + 1), loc[1]] == im[loc[0], loc[1]]).all():
                if (loc[0]+1, loc[1]) not in visited:
                    nbs.append((loc[0]+1, loc[1]))
        if loc[0] - 1 >= 0:
            if (im[loc[0] - 1, loc[1]] == im[loc[0], loc[1]]).all():
                if (loc[0] - 1, loc[1]) not in visited:
                    nbs.append((loc[0]-1, loc[1]))
        if loc[1] + 1 < width:
            if (im[loc[0], loc[1]+1] == im[loc[0], loc[1]]).all():
                if (loc[0], loc[1]+1) not in visited:
                    nbs.append((loc[0], loc[1]+1))
        if loc[1] - 1 >= 0:
            if (im[loc[0], loc[1]-1] == im[loc[0], loc[1]]).all():
                if (loc[0], loc[1]-1) not in visited:
                    nbs.append((loc[0], loc[1]-1))
        return nbs

    def bfs():
        while len(queue) > 0:
            loc = queue.pop(0)
            visited.append(loc)
            for x in neighbors(loc):
                if (x not in visited) and (x not in queue):
                    queue.append(x)
                    block.append(x)
    blocks = []
    for l in locs:
        if l not in visited:
            queue.append(l)
            bfs()
            if block:
                color = im[block[0][0], block[0][1]]
                blocks.append([color.tolist(), len(block), len(block)/(height*width)])
                block = []
    return blocks


class Distribution(object):
    @staticmethod
    def bi_value(im):
        """
        功能: 将图像二值化
            先将图片转化成灰度图像[0-255]，然后计算平均灰度值，将平均灰度值作为
            阈值
        参数:
            im: 图像 PIL Image 格式
        返回: 二值化图像 PIL Image 格式
        """
        lim = im.convert("L")
        # getting average value as threshold
        lim_ = np.array(lim)
        threshold = np.mean(lim_)
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)
        bim = lim.point(table, '1')
        # plt.imshow(bim)
        # plt.show()
        return bim

    @staticmethod
    def location(bim):
        """
        功能: 将二值化后图像中的值为1的点的位置（相对位置）记录下来
        参数:
            bim: 二值化图像 Numpy Ndarray 格式
        返回: 相对位置序列 [[离左边界的长度比, 离上边界的长度比], ...]
        """
        h = np.size(bim, axis=0)  # height
        w = np.size(bim, axis=1)  # weight
        loc = []
        for i in range(h):
            for j in range(w):
                if not bim[i, j]:
                    rel_h = i / h
                    rel_w = j / w
                    loc_ = (rel_w, rel_h)
                    loc.append(loc_)
        return loc

    def __call__(self, file):
        image = Image.open(file)
        bimage = self.bi_value(image)
        locals_ = self.location(np.array(bimage))
        random.shuffle(locals_)
        return locals_


def paint_circle(img, block, coord):
    height = img.shape[0]
    width = img.shape[1]
    c = block[0]
    area_proportion = block[2]
    r = int(np.sqrt((area_proportion * height * width / np.pi)))  # 半径
    cv.circle(img, coord, r, c, -1)
    return img


