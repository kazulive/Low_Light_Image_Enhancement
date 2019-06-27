import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def graph(x, y):
    sns.set()
    plt.plot(x, lw=2, label="L2 Norm")
    plt.plot(y, lw=2, label="Lp Norm(p=0.4)")
    plt.title("Intensity Feature", fontsize=20)
    plt.ylabel("intensity", fontsize=15)
    plt.xlabel("point", fontsize=15)
    plt.legend(loc="lower left", fontsize=15)
    plt.ylim([100, 200])
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    input_file1 = input("input file_name1 : ")
    input_file2 = input("input file_name2 : ")
    x = int(input("strat x_axis : "))
    y = int(input("start y_axis : "))
    img1 = cv2.imread(input_file1 + ".bmp")
    img2 = cv2.imread(input_file2 + ".bmp")
    H, W = img1.shape[:2]
    intensity1 = list(img1[y, x : x + W, 0])
    intensity2 = list(img2[y, x : x + W, 0])
    print(intensity1)
    graph(intensity1, intensity2)
    cv2.line(img1, (x, y), (W - 1, y), (0, 0, 255), 1)
    cv2.line(img2, (x, y), (W - 1, y), (0, 0, 255), 1)
    cv2.imshow("input_image1", img1)
    cv2.imshow("input_image2", img2)
    cv2.waitKey(0)