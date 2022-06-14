"""探知機文字部分の画像を色変化前の画像っぽく戻す:param image: 読み込んだ探知機の画像配列:return: 背景を白に戻した画像"""

import cv2
import numpy as np

def fix_lie_image(image):
    
    colors = image.reshape(-1, 3).astype(np.float32)
    # クラスタ数
    K = 15
    # # 最大反復回数: 10、移動量の閾値: 1.0
    criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
    ret, labels, centers = cv2.kmeans(
    colors, K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.squeeze(axis=1)
    centers = centers.astype(np.uint8)

    # 各クラスタに属するサンプル数を計算する。
    _, counts = np.unique(labels, axis=0, return_counts=True)
    #背景色特定(一番使われている色の平均)
    max_i = np.argmax(counts)
    max_color = list(centers[max_i])

    # 1ドットずつ色を変更
    # # 多分探知機の画像生成の処理が白背景の画像からBGRの値3色の1~3つを-254してる感じっぽいのでその逆をやる
    # # BGR値が200を超えてたらその色だけ変更してるけど、もしかしたら200以下もあるかも
    for key1, value1 in enumerate(image):
        for key2, value2 in enumerate(value1):
            for i in range(3):
                if (100 < max_color[i]):
                     #背景色 -1でやってるけど精度が悪かったら-3とかにしてみる
                    image[key1][key2][i] = abs(image[key1][key2][i] - (max_color[i]-1))
    # 反転
    image = cv2.bitwise_not(image)
    return image


# image = cv2.imread('a.PNG')


# image_converted = fix_lie_image(image)

# cv2.imwrite("image.jpeg", image_converted)