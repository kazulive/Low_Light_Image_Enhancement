import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import cv2

# limeのmain処理
class IlluminationEstimation:
    def __init__(self, alpha, norm_p, eta, scale, eps):
        self.alpha = alpha
        self.norm_p = norm_p
        self.eta = eta
        self.scale = scale
        self.eps = eps

        self.kernel = np.array([[-0, 0, 0],
                                [-1, 1, 0],
                                [-0, 0, 0]])
    def compute_weight_map(self, img):

        # ∇Iの計算
        img_h = np.hstack([np.diff(img, axis=1), (img[:, 0]-img[:, -1]).reshape(-1, 1)])
        img_v = np.vstack([np.diff(img, axis=0), (img[0, :] - img[-1, :]).reshape(1, -1)])
        # uの計算
        uh = np.ones(img.shape, dtype=np.float32) * (1. / (self.eta ** (2-self.norm_p)))
        uv = np.copy(uh)
        uh[np.abs(img_h) > self.eta] = 1.0 / ((np.abs(img_h[np.abs(img_h) > self.eta]))**(2 - self.norm_p) + 1e-3)
        uv[np.abs(img_v) > self.eta] = 1.0 / ((np.abs(img_v[np.abs(img_v) > self.eta])) ** (2 - self.norm_p) + 1e-3)

        #uh = 1.0 / (np.abs(cv2.filter2D(img, -1, self.kernel)) + 1e-3)
        #uv = 1.0 / (np.abs(cv2.filter2D(img, -1, self.kernel.T)) + 1e-3)
        return uh, uv

    def solve_linear_equation(self, img, Ih, Wx, Wy):
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
        illumination, info = scipy.sparse.linalg.bicgstab(a, Ih_vec, tol=1e-3, maxiter=2000, M=m2)

        if info != 0:
            print("収束不可能でした")

        #illumination = scipy.sparse.linalg.spsolve(a, Ih_vec, use_umfpack=True)

        illumination = illumination.reshape((H, W), order='C')

        #illumination = np.clip(illumination, 0, sys.maxsize)
        #illumination = illumination / (np.max(illumination) + self.eps)

        return illumination

    def get_illumination(self, img, illumination):#,reflectance):
        count = 0
        while(True):
            prev_illumination = np.copy(illumination)
            init_illumination = np.copy(img)
            Wx, Wy = self.compute_weight_map(illumination)
            # 照明画像を更新
            illumination = self.solve_linear_equation(img, init_illumination, Wx, Wy)
            if(count != 0):
                if(np.sum(illumination)-np.sum(prev_illumination) < 0.001):
                    break
            count += 1
        return illumination

if __name__ == '__main__':
    img = cv2.imread('./testdata/BMP/6.bmp')
    cv2.imshow("input", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    output = IlluminationEstimation(alpha=0.01, norm_p= 0.4, eta = 1./8., scale=1.0, eps=1e-3).get_illumination(v)
    #output = IlluminationEstimation(alpha=0.01, norm_p=0.4, eta=1. / 8., scale=1.0, eps=1e-3).get_illumination(v)
    #output = IlluminationEstimation(alpha=0.01, norm_p=1.0, eta=1. / 8., scale=1.0, eps=1e-3).get_illumination(v)
    output = IlluminationEstimation(alpha=0.01, norm_p=2.0, eta=1. / 8., scale=1.0, eps=1e-3).get_illumination(v)

    cv2.waitKey(0)