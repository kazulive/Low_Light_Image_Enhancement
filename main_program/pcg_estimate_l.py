import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cv2
import sys

# limeのmain処理
class IlluminationEnhnacement:
    def __init__(self, alpha, gamma, scale, eps):
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale
        self.eps = eps

    def solve_linear_equation(self, Ih, Wx, Wy):
        """
        :param Ih: 初期 I^, shape=(h, w)
        :param Wx: 式(19)によるWd(x) (horizontal), shape=(h, w)
        :param Wy: 式(19)によるWd(x) (vertical), shape=(h, w)
        """
        H, W = Ih.shape[:2]
        N = H * W

        # ベクトル化
        Ih_vec = Ih.flatten('C')

        # 式(19)はAx=b (x=t, b=t~)で表現可能
        dx = self.alpha * Wx
        dy = self.alpha * Wy
        dxa = np.hstack([dx[:, -1].reshape(-1, 1), dx[:, 0:-1]]) # dx ahead
        dya = np.vstack([dy[-1, :].reshape(1, -1), dy[0:-1, :]]) # dy ahead

        # ベクトル化
        dy_vec = dy.flatten('C')
        dx_vec = dx.flatten('C')
        dxa_vec = dxa.flatten('C')
        dya_vec = dya.flatten('C')

        dyd1 = -np.vstack([dy[-1, :].reshape(1, -1), np.zeros((H-1, W))]).flatten('C')
        dyd2 = -np.vstack([dya[1:, :], np.zeros((1, W))]).flatten('C')
        dyd3 = -np.vstack([np.zeros((1, W)), dy[0:-1, :]]).flatten('C')
        dyd4 = -np.vstack([np.zeros((H-1, W)), dya[0, :].reshape(1, -1)]).flatten('C')
        ay = scipy.sparse.spdiags(np.array([dyd1, dyd2, dyd3, dyd4]), np.array([-N+W, -W, W, N-W]), N, N)

        dxd1 = -np.hstack([dx[:, -1].reshape(-1, 1), np.zeros((H, W - 1))]).flatten('C')
        dxd2 = -np.hstack([dxa[:, 1:], np.zeros((H, 1))]).flatten('C')
        dxd3 = -np.hstack([np.zeros((H, 1)), dx[:, 0:-1]]).flatten('C')
        dxd4 = -np.hstack([np.zeros((H, W - 1)), dxa[:, 0].reshape(-1, 1)]).flatten('C')
        ax = scipy.sparse.spdiags(np.array([dxd1, dxd2, dxd3, dxd4]), np.array([-W + 1, -1, 1, W - 1]), N, N)

        dig = scipy.sparse.spdiags(np.array([dx_vec + dy_vec + dxa_vec + dya_vec + 1]), np.array([0]), N, N)
        a = ax + ay + dig

        # 逆行列Aの近似
        m = scipy.sparse.linalg.spilu(a.tocsc())
        # 線形関数を構成
        m2 = scipy.sparse.linalg.LinearOperator((N, N), m.solve)
        # 前処理付き共役勾配法
        T, info = scipy.sparse.linalg.bicgstab(a, Ih_vec, tol=1e-3, maxiter=2000, M=m2)

        if info != 0:
            print("収束不可能でした")

        T = T.reshape((H, W), order='C')

        T = np.clip(T, 0, sys.maxsize)
        T = T / (np.max(T) + self.eps)

        return T

    def get_illumination(self, img):
        # 画像サイズ確保
        H, W = img.shape[:2]

        # ダウンサンプリング
        dH = int(H * self.scale)
        dW = int(W * self.scale)
        down_img = cv2.resize(img, (dW, dH), interpolation=cv2.INTER_AREA)

        # 画素値の正規化 [0, 1]
        down_img = down_img / 255.

        # 初期照明画像の生成 I0 <- v_ch
        illumination = np.copy(down_img)
        print(illumination.shape)
        # ∇Iの計算
        illumination_h = np.hstack([np.diff(illumination, axis=1), (illumination[:, 0]-illumination[:, -1]).reshape(-1, 1)])
        illumination_v = np.vstack([np.diff(illumination, axis=0), (illumination[0, :] - illumination[-1, :]).reshape(1, -1)])


        Wx = 1.0 / (np.abs(illumination_h) + self.eps)
        Wy = 1.0 / (np.abs(illumination_v) + self.eps)

        # 照明画像を更新
        estimate_illumination = self.solve_linear_equation(illumination, Wx, Wy)
        cv2.imshow("Illumination", estimate_illumination)
        # γ関数
        estimate_illumination = np.power(estimate_illumination, 1. / self.gamma)
        estimate_illumination = np.expand_dims(estimate_illumination, axis=2) + self.eps

        cv2.imshow("Estimate_Illumination", estimate_illumination)
        cv2.waitKey(0)

        return

if __name__ == '__main__':
    img = cv2.imread('./testdata/BMP/10.bmp')
    cv2.imshow("input", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    output = IlluminationEnhnacement(alpha=0.007, gamma=2.2, scale=1.0, eps=1e-3).get_illumination(v)
