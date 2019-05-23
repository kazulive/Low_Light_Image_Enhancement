import cv2
import numpy as np
from padding import psf2otf as psf
from padding import getKernel as gK
import glob
from natsort import natsorted

class IlluminationEstimation(object):
    def __init__(self, img, alpha):
        self.alpha = alpha
        self.height = img.shape[0]
        self.width = img.shape[1]

        self.kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

        self.F_conj_h = psf(np.expand_dims(np.array([1, -1]), axis=1), (self.height, self.width)).conjugate()
        self.F_conj_v = psf(np.expand_dims(np.array([1, -1]), axis=1).T, (self.height, self.width)).conjugate()
        self.F_div = gK(img)[1]

    def get_illumination(self, img, Th, Tv, Zh, Zv, meu):
        tmp1 = np.fft.fft2(2.0 * img) + meu * self.F_conj_h * np.fft.fft2(Th - Zh/meu) + self.F_conj_v * np.fft.fft2(Tv - Zv/meu)
        tmp2 = 2.0 + meu * self.F_div
        return np.real(np.fft.ifft2(tmp1 / tmp2))

    def get_T(self, grad_h, grad_v, Zh, Zv, meu):
        tmp_h = grad_h + Zh / meu
        tmp_v = grad_v + Zv / meu
        return np.sign(tmp_h) * np.maximum(np.abs(tmp_h)-self.alpha/meu, 0.), np.sign(tmp_v) * np.maximum(np.abs(tmp_v)-self.alpha/meu, 0.)

    def get_Z(self, grad_h, grad_v, Th, Tv, Zh, Zv, meu):
        return Zh + meu * (grad_h - Th), Zv + meu * (grad_v - Tv)

    def main(self, img):
        init_illumination = cv2.GaussianBlur(img, (5, 5), 2.0)
        Th = np.zeros((self.height, self.width), dtype=np.float32)
        Tv = np.zeros((self.height, self.width), dtype=np.float32)
        Zh = np.zeros((self.height, self.width), dtype=np.float32)
        Zv = np.zeros((self.height, self.width), dtype=np.float32)
        meu = 1.0
        p = 1.5
        k = 0
        while (k < 5):
            illumination = self.get_illumination(init_illumination, Th, Tv, Zh, Zv, meu)
            grad_h = cv2.Sobel(img/255.0, cv2.CV_64F, 1, 0, ksize=3)
            grad_v = cv2.Sobel(img/255.0, cv2.CV_64F, 0, 1, ksize=3)
            Th, Tv = self.get_T(grad_h, grad_v, Zh, Zv, meu)
            Zh, Zv = self.get_Z(grad_h, grad_v, Th, Tv, Zh, Zv, meu)
            meu *= p
            k += 1
        return illumination

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
        illumination = IlluminationEstimation(v, 0.007).main(v.astype(dtype=np.float32))
        #illumination = illumination * 255.0
        #illumination = np.clip(illumination, 0, 255)
        #illumination = np.fix(illumination).astype(dtype=np.uint8)
        cv2.imshow("Illumination", illumination.astype(dtype=np.uint8))  # .astype(dtype=np.uint8))
        cv2.waitKey(0)