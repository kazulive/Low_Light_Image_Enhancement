import cv2
import numpy as np
import glob
from natsort import natsorted

# 同パッケージ内のモジュール
from padding import psf2otf as psf
from padding import getKernel as gK
from estimate_l import IlluminationEstimation
from clahe import *                 # CLAHE関数
from shrink import *                # shrinkage関数

class ReflectanceEstimation(object):
    def __init__(self, img, alpha, beta, omega, eps, lam, sigma):
        self.height, self.width = img.shape[:2]
        self.beta = beta
        self.omega = omega
        self.eps = eps
        self.lam = lam
        self.sigma = sigma

        self.kernel = np.array([[0, 0, 0],
                                [-1, 0, 1],
                                [0, 0, 0]])

        # 微分オペレータ、デルタ関数のFFT
        self.F_h = psf(np.expand_dims(np.array([1, -1]), axis=1), (self.height, self.width))
        self.F_v = psf(np.expand_dims(np.array([1, -1]), axis=1).T, (self.height, self.width))
        self.F_conj_h = psf(np.expand_dims(np.array([1, -1]), axis=1), (self.height, self.width)).conjugate()
        self.F_conj_v = psf(np.expand_dims(np.array([1, -1]), axis=1).T, (self.height, self.width)).conjugate()
        self.F_div = gK(img)[1]

        # Illuminatioを計算
        self.illumination = IlluminationEstimation(img, alpha).main(img.astype(dtype=np.float32))

        # 入力画像の正規化 [0, 1]
        dst = img.astype(dtype=np.float32) / 255.
        # 入力画像の勾配画像
        self.grad_h = cv2.filter2D(img, cv2.CV_32F, self.kernel)
        self.grad_v = cv2.filter2D(img, cv2.CV_32F, self.kernel)

    # 重み行列
    def get_weigth_matrix(self):
        # Wを計算
        Wh = 1.0 / (np.abs(self.grad_h) + self.eps)
        Wv = 1.0 / (np.abs(self.grad_v) + self.eps)
        return Wh, Wv

    # 勾配重み行列
    def get_gradient_matrix(self):
        grad_h = np.copy(self.grad_h)
        grad_v = np.copy(self.grad_v)
        # ∇S^を計算
        print(np.max(self.grad_h))
        grad_h[np.abs(grad_h) < self.eps] = 0.0
        grad_v[np.abs(grad_v) < self.eps] = 0.0
        # Gを計算
        Gh = (1.0 + self.lam * np.exp(-np.abs(grad_h)/self.sigma)) * grad_h
        Gv = (1.0 + self.lam * np.exp(-np.abs(grad_v)/self.sigma)) * grad_v
        return Gh, Gv

    def get_reflectance(self, img, Wh, Wv, Gh, Gv):
        phi = self.omega * (self.F_conj_h * np.fft.fft2(Gh) + self.F_conj_v * np.fft.fft2(Gv))
        up = np.fft.fft2(img / self.illumination)

        tmp = self.beta * (self.F_conj_h * np.fft.fft2(Wh) * self.F_h + self.F_conj_v * np.fft.fft2(Wv) * self.F_v)
        bottom = 1.0 + tmp + self.omega * self.F_div

        return np.real(np.fft.ifft2(up / bottom))

    def main(self, img):
        Wh, Wv = self.get_weigth_matrix()
        Gh, Gv = self.get_gradient_matrix()
        reflectance = self.get_reflectance(img, Wh, Wv, Gh, Gv)
        illumination = np.copy(self.illumination)
        cv2.imshow("Reflectance", reflectance)
        cv2.imshow("Illumination", illumination.astype(dtype = np.uint8))

        return reflectance, illumination

def gamma_correction(img, gamma):
    output = (img.astype(dtype=np.float32) / 255.0) ** (1. / gamma) * 255.
    output = np.clip(output, 0, 255)
    output = np.fix(output).astype(dtype=np.uint8)
    return output

def fileRead():
    data = []
    for file in natsorted(glob.glob('./testdata/BMP/*.bmp')):
        data.append(cv2.imread(file, 1))
    return data

if __name__ == '__main__':
    img_list = fileRead()
    count = 0
    for img in img_list:
        count += 1
        print('input ' + str(count) + ' image')
        # HSV変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        reflectance, illumination = ReflectanceEstimation(v.astype(dtype=np.float32), 0.007, 0.001, 0.016, 10, 6, 10).main(v.astype(dtype=np.float32))
        # illumination = illumination * 255.0
        # illumination = np.clip(illumination, 0, 255)
        # illumination = np.fix(illumination).astype(dtype=np.uint8)
        result = reflectance * gamma_correction(illumination, 2.2)
        result = np.clip(result, 0.0, 255.0)
        result = np.fix(result).astype(dtype=np.uint8)
        output = cv2.merge((h, s, result))
        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        cv2.imshow("output", output)
        cv2.waitKey(0)