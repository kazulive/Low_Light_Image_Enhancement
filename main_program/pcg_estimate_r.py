"""
Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T, np.array([-N + H, -H]), N,N)
        Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0), np.array([-H + 1, -1]), N, N)
        D = 1 - (dx + dy + dxa + dya)
        A = ((Ax + Ay) + (Ax + Ay).conj().T + scipy.sparse.spdiags(D, 0, N, N)).T

        main_diag = np.ones(N)*-4.0
        side_diag = np.ones(N-1)
        up_down_diag = np.ones(N-3)
        diagonals = [main_diag, side_diag, side_diag, up_down_diag,up_down_diag]
        laplacian = self.omega * scipy.sparse.spdiags(diagonals, [0, -1, 1, H, -H], N, N)
"""

import numpy as np
import sys
import scipy.sparse
import scipy.sparse.linalg
import cv2
import glob
from natsort import natsorted

from pcg_estimate_l import IlluminationEstimation

class ReflectanceEstimation:
    def __init__(self, img, illumination, beta, omega, eps, lam, sigma):
        self.beta = beta
        self.omega = omega
        self.eps = eps
        self.lam = lam
        self.sigma = sigma

        self.kernel = np.array([[0, 0, 0],
                                [-1, 1, 0],
                                [0, 0, 0]])
        # Illuminatioを計算
        self.illumination = illumination

        dst = img * 255.0
        # 入力画像の勾配画像
        self.grad_h = cv2.filter2D(dst, cv2.CV_32F, self.kernel)
        self.grad_v = cv2.filter2D(dst, cv2.CV_32F, self.kernel)

    # 勾配重み行列
    def compute_gradient_map(self):
        grad_h = np.copy(self.grad_h)
        grad_v = np.copy(self.grad_v)
        grad_h = np.clip(grad_h, 0.0, 255.0)
        grad_v = np.clip(grad_v, 0.0, 255.0)
        # ∇S^を計算
        grad_h[np.abs(grad_h) < self.eps] = 0.0
        grad_v[np.abs(grad_v) < self.eps] = 0.0
        # Gを計算
        Gh = (1.0 + self.lam / (np.exp(np.abs(grad_h) / self.sigma))) * grad_h
        Gv = (1.0 + self.lam / (np.exp(np.abs(grad_v) / self.sigma))) * grad_v
        cv2.imshow("Gv", Gh)
        cv2.imshow("Gv", Gv)
        return Gh, Gv

    # 重み行列
    def compute_weigth_map(self):
        # Wを計算
        Wh = 1.0 / (np.abs(self.grad_h) + 1e-3)
        Wv = 1.0 / (np.abs(self.grad_v) + 1e-3)
        cv2.imshow("WH", Wh)

        return Wh, Wv

    def solve_linear_equation(self, img, illumination, Wh, Wv, Gh, Gv):

        H, W = img.shape[:2]
        N = H * W

        # ベクトル化
        dx = -Wh.flatten('F')
        dy = -Wv.flatten('F')
        tempx = np.roll(Wh, 1, axis=1)
        tempy = np.roll(Wv, 1, axis=0)
        dxa = -tempx.flatten('F')
        dya = -tempy.flatten('F')
        tmp = Wh[:, -1]
        tempx = np.concatenate((tmp[:,None], np.zeros((H, W-1))), axis=1)
        tmp = Wv[-1,:]
        tempy = np.concatenate((tmp[None, :], np.zeros((H-1, W))), axis=0)
        dxd1 = -tempx.flatten('F')
        dyd1 = -tempy.flatten('F')

        Wh[:,-1] = 0
        Wv[-1,:] = 0
        dxd2 = -Wh.flatten('F')
        dyd2 = -Wv.flatten('F')

        Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T, np.array([-N + H, -H]), N,
                                  N)
        Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0), np.array([-H + 1, -1]), N, N)
        D = 1 - (dx + dy + dxa + dya)
        A = ((Ax + Ay) + (Ax + Ay).conj().T + scipy.sparse.spdiags(D, 0, N, N)).T

        main_diag = np.ones(N) * 4.0
        side_diag = np.ones(N - 1) * -1.0
        diagonals = [main_diag, side_diag, side_diag]
        laplacian = self.omega * scipy.sparse.diags(diagonals, [0, -1, 1], format="csr")

        A += laplacian

        tin = img.flatten('F') / (illumination.flatten('F') + 1e-3)

        # ベクトル化
        gh = -Gh.flatten('F')
        gv = -Gv.flatten('F')

        gh[1:N] = gh[1:N] + gh[0:N-1]
        gv[0:N-1] = gv[0:N-1] - gv[1:N]
        tin = tin + self.omega * (gh + gv)
        # 逆行列Aの近似
        m = scipy.sparse.linalg.spilu(A.tocsc())
        # 線形関数を構成
        m2 = scipy.sparse.linalg.LinearOperator((N, N), m.solve)
        # 前処理付き共役勾配法
        tout, info = scipy.sparse.linalg.bicgstab(A, tin, tol=1e-2, maxiter=2000, M=m2)
        #tout = scipy.sparse.linalg.spsolve(A, tin)
        OUT = np.reshape(tout, (H, W), order='F')
        OUT = np.clip(OUT, 0.0, sys.maxsize)
        OUT = OUT / (np.max(OUT) + 1e-3)
        return OUT

    def get_reflectance(self, img, illumination):
        Wx, Wy = self.compute_weigth_map()
        Gx, Gy = self.compute_gradient_map()
        estimate_reflectance = self.solve_linear_equation(img, illumination, Wx, Wy, Gx, Gy)
        return estimate_reflectance

def gamma_correction(img, gamma):
    output = (img.astype(dtype=np.float32)) ** (1. / gamma)
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
        img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # BGR分割
        b, g, r = cv2.split(img)
        # HSV変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 輝度画像を用いて照明画像を推定
        illumination = IlluminationEstimation(alpha=0.001, norm_p= 0.1, eta = 1./8., scale=1.0, eps=1e-3).get_illumination(v.astype(dtype=np.float32))
        #illumination = np.minimum(1.0, np.maximum(illumination, 0.0))
        # BGRそれぞれで反射画像を生成
        b_reflectance = ReflectanceEstimation(b, illumination, beta=0.001, omega=0.016, eps=10.0, lam=6.0, sigma=10.0).get_reflectance(b, illumination)
        g_reflectance = ReflectanceEstimation(g, illumination, beta=0.001, omega=0.016, eps=10.0, lam=6.0, sigma=10.0).get_reflectance(g, illumination)
        r_reflectance = ReflectanceEstimation(r, illumination, beta=0.001, omega=0.016, eps=10.0, lam=6.0, sigma=10.0).get_reflectance(r, illumination)
        reflectance = cv2.merge((b_reflectance, g_reflectance, r_reflectance))
        b_result = b_reflectance * gamma_correction(illumination, 2.2)
        g_result = g_reflectance * gamma_correction(illumination, 2.2)
        r_result = r_reflectance * gamma_correction(illumination, 2.2)
        result = cv2.merge((b_result, g_result, r_result))
        #result = np.clip(result, 0.0, 255.0)
        #result = np.fix(result).astype(dtype=np.uint8)
        cv2.imshow("Illumination", illumination)
        cv2.imshow("Reflectance", reflectance)
        cv2.imshow("Output", result)
        cv2.waitKey(0)