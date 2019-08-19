import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cv2
import glob
from natsort import natsorted

from pcg_estimate_l import IlluminationEstimation
from utils import Visualization

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

        # 入力画像の勾配画像
        self.grad_h = cv2.filter2D(img, -1, self.kernel)
        self.grad_v = cv2.filter2D(img, -1, self.kernel.T)

    # 勾配重み行列
    def compute_gradient_map(self):
        grad_h = np.copy(self.grad_h) * 255.0
        grad_v = np.copy(self.grad_v) * 255.0

        grad_h = np.clip(grad_h, 0.0, 255.0)
        grad_v = np.clip(grad_v, 0.0, 255.0)


        # 最大・最小閾値を計算
        grad_h_min = np.min(grad_h) / 2.0
        grad_h_max = (np.max(grad_h) + 2.0 * np.mean(grad_h)) / 3.0
        grad_v_min = np.min(grad_v) / 2.0
        grad_v_max = (np.max(grad_v) + 2.0 * np.mean(grad_v)) / 3.0

        # ∇S^を計算
        grad_h[np.abs(grad_h) < grad_h_min] = grad_h_min
        grad_v[np.abs(grad_v) < grad_v_min] = grad_v_min
        grad_h[np.abs(grad_h) > grad_h_max] = grad_h_max
        grad_v[np.abs(grad_v) > grad_v_max] = grad_v_max

        # Gを計算
        Gh = (1.0 + self.lam * (np.exp(-1 * np.abs(grad_h) / self.sigma))) * grad_h
        Gv = (1.0 + self.lam * (np.exp(-1 * np.abs(grad_v) / self.sigma))) * grad_v

        Gh = np.clip(Gh, 0.0, 255.0) / 255.0
        Gv = np.clip(Gv, 0.0, 255.0) / 255.0

        return Gh, Gv

    # 重み行列
    def compute_weigth_map(self, img, sigma1, sigma2):

        dx = cv2.filter2D(img, -1, self.kernel) #* 255.0
        dy = cv2.filter2D(img, -1, self.kernel.T) #* 255.0

        gdx1 = cv2.GaussianBlur(dx, (3, 3), sigma1)
        gdy1 = cv2.GaussianBlur(dy, (3, 3), sigma1)
        gdx2 = cv2.GaussianBlur(dx, (3, 3), sigma2)
        gdy2 = cv2.GaussianBlur(dy, (3, 3), sigma2)

        Wh = np.maximum(np.abs(gdx1) * np.abs(gdx2), self.eps) ** (-1)
        Wv = np.maximum(np.abs(gdy1) * np.abs(gdy2), self.eps) ** (-1)
        Wh = cv2.GaussianBlur(Wh, (3, 3), sigma1 / 2.0)
        Wv = cv2.GaussianBlur(Wv, (3, 3), sigma1 / 2.0)

        Wh[:, -1] = 0.0
        Wv[-1, :] = 0.0

        #Wh = np.zeros(img.shape)
        #Wv = Wh
        """
        dx = cv2.filter2D(img, -1, self.kernel) * 255.0
        dy = cv2.filter2D(img, -1, self.kernel.T) * 255.0
        Wh = 1.0 / (np.abs(dx) + 1e-3)
        Wv = 1.0 / (np.abs(dy) + 1e-3)
        """
        return Wh, Wv

    def solve_linear_equation(self, img, illumination, Wh, Wv, Gh, Gv):

        H, W = img.shape[:2]
        N = H * W

        # ベクトル化
        dx = -self.beta * Wh.flatten('F')
        dy = -self.beta * Wv.flatten('F')
        tempx = np.roll(Wh, 1, axis=1)
        tempy = np.roll(Wv, 1, axis=0)
        dxa = -self.beta * tempx.flatten('F')
        dya = -self.beta * tempy.flatten('F')
        tmp = Wh[:, -1]
        tempx = np.concatenate((tmp[:,None], np.zeros((H, W-1))), axis=1)
        tmp = Wv[-1,:]
        tempy = np.concatenate((tmp[None, :], np.zeros((H-1, W))), axis=0)
        dxd1 = -self.beta * tempx.flatten('F')
        dyd1 = -self.beta * tempy.flatten('F')

        Wh[:,-1] = 0
        Wv[-1,:] = 0
        dxd2 = -self.beta * Wh.flatten('F')
        dyd2 = -self.beta * Wv.flatten('F')

        Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:, None], dxd2[:, None]), axis=1).T, np.array([-N + H, -H]), N,
                                  N)
        Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None, :], dyd2[None, :]), axis=0), np.array([-H + 1, -1]), N, N)
        D = 1 - (dx + dy + dxa + dya)
        A = ((Ax + Ay) + (Ax + Ay).conj().T + scipy.sparse.spdiags(D, 0, N, N))

        main_diag = np.ones(N) * 4.0
        main_diag[0] = main_diag[N-1] = 3.0
        side_diag = np.ones(N - 1) * -2.0
        side2_diag = np.ones(N - 2) * -1.0
        diagonals = [main_diag, side_diag, side_diag, side2_diag, side2_diag]
        laplacian = self.omega * scipy.sparse.diags(diagonals, [0, -1, 1, -2, 2], format="csr")

        A = A + laplacian

        tin = ((img*255.0) / (illumination*255.0 + 1e-3)).flatten('F')

        # ベクトル化
        gh = -self.omega * Gh.flatten('F')
        gv = -self.omega * Gv.flatten('F')

        #gh[0 : N-1] = -gh[0 : N-1] + gh[1 : N]
        #gv[1 : N] = -gv[1 : N] + gv[0 : N-1]
        gh[1:N] =  gh[1 : N] - gh[0 : N-1]
        gv[0:N-1] = gv[0 : N-1] - gv[1 : N]
        #gh[0] = 0
        #gv[-1] = 0
        tin = tin + (gh + gv)

        # 逆行列Aの近似
        m = scipy.sparse.linalg.spilu(A.tocsc())
        # 線形関数を構成
        m2 = scipy.sparse.linalg.LinearOperator((N, N), m.solve)
        # 前処理付き共役勾配法
        tout, info = scipy.sparse.linalg.bicgstab(A, tin, tol=1e-5, maxiter=3000, M=m2)
        #tout = scipy.sparse.linalg.spsolve(A, tin)
        OUT = np.reshape(tout, (H, W), order='F')
        return OUT

    def get_reflectance(self, input, img, illumination):
        count = 0
        Wx, Wy = self.compute_weigth_map(img, 1.0, 3.0)
        """
        while(count < 3):
            if(count != 0):
                Wx, Wy = self.compute_weigth_map(reflectance, 1.0, 3.0)
            Gx, Gy = self.compute_gradient_map()
            reflectance = self.solve_linear_equation(img, illumination, Wx, Wy, Gx, Gy)
            count += 1
        """
        Gx, Gy = self.compute_gradient_map()
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
    for file in natsorted(glob.glob('./testdata/Image/*.bmp')):
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
        # YCrCb変換
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(YCrCb)
        #Visualization(dir_name=False, file_name="Y_show", flag=0).show(Y)
        #illumination = np.copy(Y)
        # 照明画像を計算
        illumination = IlluminationEstimation(alpha=0.01, norm_p=0.4, eta=1. / 8., scale=1.0,
                                              eps=1e-3).get_illumination(Y)#, illumination)
        print("Extract Illumination Image")

        # 反射画像を生成
        b_reflectance, b_Wx, b_Wy, b_Gx, b_Gy = ReflectanceEstimation(b, illumination, beta=0.001, omega=0.016, eps=0.0001, lam=10.0, sigma=10.0).get_reflectance(img, b, illumination)
        print("Extract B_Reflectance Image")
        g_reflectance, g_Wx, g_Wy, g_Gx, g_Gy= ReflectanceEstimation(g, illumination, beta=0.001, omega=0.016, eps=0.0001, lam=10.0, sigma=10.0).get_reflectance(img, g, illumination)
        print("Extract G_Reflectance Image")
        r_reflectance, r_Wx, r_Wy, r_Gx, r_Gy= ReflectanceEstimation(r, illumination, beta=0.001, omega=0.016, eps=0.0001, lam=10.0, sigma=10.0).get_reflectance(img, r, illumination)
        print("Extract R_Reflectance Image")

        reflectance = cv2.merge((b_reflectance, g_reflectance, r_reflectance)).astype(dtype=np.float32)

        b_result = b_reflectance * gamma_correction(illumination, 2.2)
        g_result = g_reflectance * gamma_correction(illumination, 2.2)
        r_result = r_reflectance * gamma_correction(illumination, 2.2)

        Wx = cv2.merge((b_Wx, g_Wx, r_Wx))
        Wy = cv2.merge((b_Wy, g_Wy, r_Wy))
        Gx = cv2.merge((b_Gx, g_Gx, r_Gx))
        Gy = cv2.merge((b_Gy, g_Gy, r_Gy))
        result = cv2.merge((b_result, g_result, r_result))

        Visualization(dir_name="./result/Wx", file_name= str(count) + "0", flag=1).save(Wx)
        Visualization(dir_name="./result/Wy", file_name=str(count) + "0", flag=1).save(Wy)
        Visualization(dir_name="./result/Gx", file_name=str(count) + "0", flag=0).save(Gx)
        Visualization(dir_name="./result/Gy", file_name=str(count) + "0", flag=0).save(Gy)
        Visualization(dir_name="./result/illumination", file_name=str(count) + "0", flag=0).save(illumination)
        Visualization(dir_name="./result/shadow-up", file_name=str(count) + "0", flag=0).save(gamma_correction(illumination, 2.2))
        Visualization(dir_name="./result/reflectance", file_name=str(count) + "0", flag=0).save(reflectance)
        Visualization(dir_name="./result/output", file_name=str(count) + "0", flag=0).save(result)
