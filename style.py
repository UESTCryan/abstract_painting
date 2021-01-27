import cv2 as cv
import numpy as np
import random
import utils


def style_1(img, blocks, block_num, width, height, loc_distribution):
    """
    功能: 根据规则在画布上画图形
    参数:
        img: 画布 Numpy ndarray 格式
        blocks: 提取的色块 [[ 颜色[r, g, b], 大小, 占图像整体比例],  ...]
        block_num: 色块数量 int
        width: 画布宽度
        height: 画布高度
        loc_distribution: 参考分布 [[x1, y1], [x2, y2], ...]
    返回: 重构后的图像 Numpy ndarray 格式
    """
    # 先画线 再画圆
    count_line = 0  # 统计线的条数
    for k in range(block_num):
        # ---------------------------------------------
        # 如果画的是线：此坐标作为其质心的坐标
        x = int(width * loc_distribution[k][0])  # x坐标
        y = int(height * loc_distribution[k][1])  # y坐标

        area_proportion = blocks[k][2]  # 色块占比

        if area_proportion <= 0.5e-3:
            # ---------------------------------------------
            # 画线
            # 画线过程中，以（x,y）为质心，长短由原始面积决定，旋转角度随机[0, 180]
            # 但在画先过程中发现  线太多使画面显得乱 所以限制10条内
            if count_line < 11:
                start, end = [0, 0], [0, 0]
                # 旋转线条
                arc = np.pi * (random.randint(0, 181) / 180)
                # 把线条延长到边际（美观）
                # r = area_proportion * height * width / 2
                r = 1000
                dx = int(r * np.cos(arc))
                dy = int(r * np.sin(arc))
                start[0] = (x - dx) if (x - dx) >= 0 else 0
                start[1] = (y - dy) if (y - dy) >= 0 else 0
                end[0] = (x + dx) if (x + dx) <= width else width
                end[1] = (y + dy) if (y + dy) <= height else height
                start = tuple(start)
                end = tuple(end)
                # start = (int(x - area_proportion * height * width / 2), y)
                # end = (int(x + area_proportion * height * width / 2), y)
                # color
                c = blocks[k][0]
                cv.line(img, start, end, c, 1)
            count_line += 1

    for l in range(block_num):
        # ---------------------------------------------
        # 如果画的是圆：此坐标作为其中心点的坐标
        x = int(width * loc_distribution[l][0])  # x坐标
        y = int(height * loc_distribution[l][1])  # y坐标

        area_proportion = blocks[l][2]  # 色块占比

        if area_proportion > 0.5e-3:
            # ---------------------------------------------
            # 画圆
            zoom = 0.8  # 缩放比例
            r = int(np.sqrt((area_proportion * height * width / zoom / np.pi)))  # 半径
            # #防止图形触边
            x, y = utils.stay_away_from_edge(height, width, x, y, r)
            # color
            c = blocks[l][0]
            cv.circle(img, (x, y), r, c, -1)
            # ---------------------------------------------
    return img
