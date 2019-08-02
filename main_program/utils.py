# -*- coding : utf-8 -*-
import cv2
import numpy as np


class Visualization(object):
    def __init__(self, dir_name, file_name, flag):
        """
        :param dir_name: 保存するディレクトリ先
        :param file_name: 保存/表示するファイルの名前
        :param flag: [0,1], [0, 255]で形式を変える
        """
        self.dir_name = dir_name
        self.file_name = file_name
        self.flag = flag

    # 正規化関数
    def normalize(self, img):
        # [0, 1] → [0, 255]に調整
        if self.flag == 0:
            dst = img * 255.0
            dst = np.clip(dst, 0.0, 255.0)
            dst = np.fix(dst).astype(dtype=np.uint8)
            return dst
        return img

    # 保存関数
    def save(self, img):
        cv2.imwrite(str(self.dir_name) + '/' +str(self.file_name) + '.bmp', self.normalize(img))
        return

    # 表示関数
    def show(self, img):
        cv2.imshow(str(self.file_name), self.normalize(img))
        cv2.waitKey(0)
        return
