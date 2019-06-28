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

        self.kernel = np.array([[-0, 0, 0],
                                [-1, 1, 0],
                                [-0, 0, 0]])
        # Illuminatioを計算
        self.illumination = illumination

        dst = img * 255.0
        dst = np.clip(dst, 0.0, 255.0)
        # 入力画像の勾配画像
        self.grad_h = cv2.filter2D(img, -1, self.kernel)
        self.grad_v = cv2.filter2D(img, -1, self.kernel.T)

    # 勾配重み行列
    def compute_gradient_map(self, img):
        #tmp = np.exp(np.log(img * 255.0 + 1.0) - np.log(illumination * 255.0 + 1.0))
        #cv2.imshow("initial reflectance", np.exp(tmp))
        #cv2.waitKey(0)
        #print(np.max(tmp))
        #Gh = cv2.filter2D(tmp, -1, self.kernel) * 255.0
        #Gv = cv2.filter2D(tmp, -1, self.kernel.T) * 255.0
        grad_h = np.copy(self.grad_h)
        grad_v = np.copy(self.grad_v)
        # ∇S^を計算
        grad_h[np.abs(grad_h) < self.eps / 255.0] = 0.0
        grad_v[np.abs(grad_v) < self.eps / 255.0] = 0.0
        # Gを計算
        Gh = (1.0 + self.lam * (np.exp(-1 * np.abs(grad_h) / self.sigma))) * grad_h
        Gv = (1.0 + self.lam * (np.exp(-1 * np.abs(grad_v) / self.sigma))) * grad_v
        #Gh /= (np.max(Gh) + 1.0)
        #Gv /= (np.max(Gv) + 1.0)
        return Gh, Gv

    # 重み行列
    def compute_weigth_map(self, img):
        """
        gaussian = cv2.GaussianBlur(img * 255.0, ksize=(5, 5), sigmaX=2.0)
        grad_h = cv2.filter2D(gaussian, -1, self.kernel)
        grad_v = cv2.filter2D(gaussian, -1, self.kernel.T)
        # Wを計算
        Wh = 1.0 / (np.abs(grad_h) + 0.0001)
        Wv = 1.0 / (np.abs(grad_v) + 0.0001)
        #Wh = 1.0 + 1.0 / (1 + np.abs(grad_h))
        #Wv = 1.0 + 1.0 / (1 + np.abs(grad_v))
        """
        log_img = np.log10(img * 255.0 + 1.0)
        grad_h = cv2.filter2D(log_img, -1, self.kernel)
        g1 = cv2.GaussianBlur(np.abs(grad_h), (3, 3), 1.8)
        g2 = cv2.GaussianBlur(np.abs(grad_h), (3, 3), 4.0 * 1.8)
        dog = g2 - g1
        dog[dog < 0.0] = 0.0
        dog /= np.sum(dog)
        print(np.max(dog))
        cv2.imshow("DoG Filter", (dog * 255.0).astype(np.uint8))
        cv2.waitKey(0)
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


        main_diag = np.ones(N) * -4.0
        side_diag = np.ones(N - 1) * 2.0
        diagonals = [main_diag, side_diag, side_diag]
        laplacian = self.omega * scipy.sparse.diags(diagonals, [0, -1, 1], format="csr")

        A += laplacian

        tin = img.flatten('F') / (illumination.flatten('F') + 0.0001)
        print(np.max(tin))

        # ベクトル化
        gh = -self.omega * Gh.flatten('F')
        gv = -self.omega * Gv.flatten('F')

        gh[1:N] = gh[1:N] - gh[0:N-1]
        gv[0:N-1] = gv[0:N-1] - gv[1:N]
        tin = tin + (gh + gv)

        print(np.max(tin))
        # 逆行列Aの近似
        m = scipy.sparse.linalg.spilu(A.tocsc())
        # 線形関数を構成
        m2 = scipy.sparse.linalg.LinearOperator((N, N), m.solve)
        # 前処理付き共役勾配法
        tout, info = scipy.sparse.linalg.bicgstab(A, tin, tol=1e-5, maxiter=2000, M=m2)
        #tout = scipy.sparse.linalg.spsolve(A, tin)
        OUT = np.reshape(tout, (H, W), order='F')
        #OUT = np.clip(OUT, 0.0, sys.maxsize)
        #OUT = OUT / (np.max(OUT) + 1e-3)
        return OUT

    def get_reflectance(self, img, illumination):
        Wx, Wy = self.compute_weigth_map(img)
        Gx, Gy = self.compute_gradient_map(img)
        reflectance = self.solve_linear_equation(img, illumination, Wx, Wy, Gx, Gy)
        return reflectance, Wx, Wy, Gx, Gy

def gamma_correction(img, gamma):
    output = (img.astype(dtype=np.float32)) ** (1. / gamma)
    return output

def normalize(img):
    img *= 255.0
    img = np.clip(img, 0.0, 255.0)
    img = np.fix(img).astype(dtype=np.uint8)
    return img

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
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(YCrCb)
        # 輝度画像を用いて照明画像を推定
        illumination = IlluminationEstimation(alpha=0.007, norm_p= 0.4, eta = 1./8., scale=1.0, eps=1e-3, pyr_num=1).get_illumination(Y)
        print("Extract Illumination Image")
        b_reflectance, b_Wx, b_Wy, b_Gx, b_Gy = ReflectanceEstimation(b, illumination, beta=0.001, omega=0.016, eps=10.0, lam=6.0, sigma=10.0).get_reflectance(b, illumination)
        g_reflectance, g_Wx, g_Wy, g_Gx, g_Gy= ReflectanceEstimation(g, illumination, beta=0.001, omega=0.016, eps=10.0, lam=6.0, sigma=10.0).get_reflectance(g, illumination)
        r_reflectance, r_Wx, r_Wy, r_Gx, r_Gy= ReflectanceEstimation(r, illumination, beta=0.001, omega=0.016, eps=10.0, lam=6.0, sigma=10.0).get_reflectance(r, illumination)
        #normal_reflectance = cv2.merge((b/illumination, g/illumination, r/illumination))
        #cv2.imshow("normal_reflectance", normal_reflectance)
        #cv2.waitKey(0)
        reflectance = cv2.merge((b_reflectance, g_reflectance, r_reflectance))
        b_result = b_reflectance * gamma_correction(illumination, 2.2)
        g_result = g_reflectance * gamma_correction(illumination, 2.2)
        r_result = r_reflectance * gamma_correction(illumination, 2.2)
        Wx = cv2.merge((b_Wx, g_Wx, r_Wx))
        Wy = cv2.merge((b_Wy, g_Wy, r_Wy))
        Gx = cv2.merge((b_Gx, g_Gx, r_Gx))
        Gy = cv2.merge((b_Gy, g_Gy, r_Gy))
        result = cv2.merge((b_result, g_result, r_result))
        cv2.imwrite("./result/Wx/0" + str(count) + ".bmp", normalize(Wx))
        cv2.imwrite("./result/Wy/0" + str(count) + ".bmp", normalize(Wy))
        cv2.imwrite("./result/Gx/0" + str(count) + ".bmp", Gx)#normalize(Gx))
        cv2.imwrite("./result/Gy/0" + str(count) + ".bmp", Gy)#normalize(Gy))
        cv2.imwrite("./result/illumination/0"+str(count)+".bmp", normalize(illumination))
        cv2.imwrite("./result/shadow-up/0" + str(count) + ".bmp", cv2.normalize(gamma_correction(illumination, 2.2), None, 0.0, 255.0, cv2.NORM_MINMAX))
        cv2.imwrite("./result/reflectance/0" + str(count) + ".bmp", normalize(reflectance))
        cv2.imwrite("./result/output/0" + str(count) + ".bmp", normalize(result))
        cv2.waitKey(0)
        """
        cv2.imshow("Illumination", normalize(illumination))
        cv2.imshow("Reflectance", normalize(reflectance))
        cv2.imshow("Output", normalize(result))
        cv2.waitKey(0)
        """