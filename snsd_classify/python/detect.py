# -*- coding: utf-8 -*-
import cv2
import sys
import os.path
import caffe
from caffe.proto import caffe_pb2
import numpy as np
from datetime import datetime
import numpy as np
import copy
import random
import sys
import matplotlib.pyplot as plt


def label(img):
    # フレームをHSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 取得する色の範囲を指定する
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([25, 255, 255])
        # 指定した色に基づいたマスク画像の生成
    img_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # フレーム画像とマスク画像の共通の領域を抽出する。
    img_color = cv2.bitwise_and(img, img, mask=img_mask)
    # グレースケールに変換
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("a.png", img_color)
    # cv2.imshow("a.png", img_color)
        # 二値変換
    thresh = 100
    max_pixel = 255
    ret, img_dst = cv2.threshold(img_gray,thresh,max_pixel,cv2.THRESH_BINARY)

    neiborhood4 = np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]],np.uint8)
        # 8近傍の定義
    neiborhood8 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]],np.uint8)

        # 8近傍で膨張処理
    img_dilation = cv2.dilate(img_color,neiborhood8,iterations=10)

     # 8近傍で縮小処理
    img_erosion = cv2.erode(img_dilation,neiborhood8,iterations=12)

#ラベリング
    height, width, channels = img_erosion.shape[:3]
    dst = copy.copy(img_erosion)
    gray = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)
    ret, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite("bin.png",bin)
    output = cv2.connectedComponentsWithStats(bin, 8, cv2.CV_32S)
    nLabels = output[0]
    labelImage = output[1]
    status = list(output[2])
    del status[0]

    status.sort(key=lambda x:x[4],reverse=True)

    return status

def filenamelist(src_directory):
    # サブディレクトリを想定
    file_array = []
    for file in fild_all_files(src_directory):
        # .pngか.jpgのときリストに格納
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            file_array.append(file)

    return file_array

def fild_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

def create_directory(output_directory):
    if os.path.isdir(output_directory) == 0:
        print "Not exist \"%s\" folder. So create it." % output_directory
        os.makedirs(output_directory)
    else:
        print "Exist \"%s\" folder." % output_directory

if __name__ == "__main__":
    argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    argc = len(argvs)  # 引数の個数
    num =100 
    print(argvs[1])

    # まず元画像があるディレクトリを持ってくる
    image_path = "./"+argvs[1]
    # file_namesにディレクトリ込みのファイルネームを格納
    file_names = filenamelist(image_path)
    image_dir = "./" +argvs[1]+"_detect"
    create_directory(image_dir)
    for file_name in file_names:
	    in_image =  cv2.imread(file_name)
	    i=1;
	    kanban = label(in_image)
            name = file_name.split("/")
            name_out = name[2].split(".")
	    print(name_out[0])
	    for (x, y, w, h ,area) in kanban:
	    	dst=in_image[y:y+h,x:x+w]
		img_name = "./" +argvs[1]+"_detect"+"/"+ name_out[0] +"_"+str(i) + '.png'
		cv2.imwrite(img_name,dst)
                print(img_name)
		i=i+1;
