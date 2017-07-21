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
class NameRGB:
    def __init__(self,name,B,G,R):
        self.name = name
        self.B = B
        self.G = G
        self.R = R


def input_arg(argvs, argc):
    if (argc < 2 or 3 < argc):   # 引数が足りない場合は、その旨を表示
        print 'Usage: # python %s Input_filename Output_filename' % argvs[0]
        quit()        # プログラムの終了
    elif (argc == 2):
        d = datetime.now()
        filename = "img-" + d.strftime('%Y') + "-" + d.strftime('%m') + "-" + d.strftime('%d') + "-" + d.strftime('%H') + "h" + d.strftime('%M') + "m" + d.strftime('%S')  + "s.jpg"
        argvs.append(filename)

    print 'Input filename = %s' % argvs[1]
    print 'Output filename = %s' % argvs[2]
    # 引数でとったディレクトリの文字列をリターン
    return argvs



def detect(frame):
    # メンバーの名前と矩形の色定義
    MemberList = []
    MemberList.append(NameRGB("ETC",117, 163, 27))
    MemberList.append(NameRGB("kanban",27, 26, 253))

    faces  = frame
    size =(64,64)
    x=0
    y=0
    image = frame
    image = cv2.resize(image,size)
    cv2.imwrite("face.png", image)
    os.system("convert face.png -equalize face.png")
    image = caffe.io.load_image('face.png')
    predictions = classifier.predict([image], oversample=False)
    pred = np.argmax(predictions)
    sorted_prediction_ind = sorted(range(len(predictions[0])),key=lambda x:predictions[0][x],reverse=True)
    first = MemberList[sorted_prediction_ind[0]].name + " " + str(int(predictions[0,sorted_prediction_ind[0]]*100,)) + "%"

    second = MemberList[sorted_prediction_ind[1]].name + " " + str(round(predictions[0,sorted_prediction_ind[1]]*100,1)) + "%"



    print 	'first: '+str(MemberList[sorted_prediction_ind[0]].name)+' '+str(predictions[0,sorted_prediction_ind[0]]*100) + "%"

    print 	'second: '+str(MemberList[sorted_prediction_ind[1]].name)+' '+str(predictions[0,sorted_prediction_ind[1]]*100) + "%"

    for i, value in enumerate(MemberList):
        if pred == 0:
         #   print "Skip ETC!"

            
            cv2.putText(frame,first,(x, y  + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200, 200, 0),2)
            cv2.putText(frame,second,(x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200, 200, 0),2)

        elif pred == i:
            
            cv2.putText(frame,first,(x, y  +10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(value.B, value.G, value.R),2)
            cv2.putText(frame,second,(x, y +25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(value.B, value.G, value.R),2)


    return frame

# 描画関数
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]),
               (0, padsize),
               (0, padsize)) + ((0, 0),
                                ) * (data.ndim - 3)
    data = np.pad(
        data, padding, mode='constant', constant_values=(padval, padval))

    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data


if __name__ == "__main__":
    argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    argc = len(argvs)  # 引数の個数

    filepath = input_arg(argvs, argc)
    in_image = cv2.imread(filepath[1])
    out_image = filepath[2]
    mean_blob = caffe_pb2.BlobProto()
    with open('../snsd_mean.binaryproto') as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(
    mean_blob.data,
    dtype=np.float32).reshape(
        (mean_blob.channels,
        mean_blob.height,
        mean_blob.width))
    classifier = caffe.Classifier(
        '../snsd_cifar10_full.prototxt',
        '../snsd_cifar10_full_iter_1000.caffemodel',
        mean=mean_array,
        raw_scale=255)

    frame = detect(in_image)

    cv2.imshow("RGB",frame)
		#保存先ディレクトリと保存名を指定
    cv2.imwrite("out_image.png", frame)
    # キー入力待機
    
	
    print [(k, v.data.shape) for k, v in classifier.blobs.items()]
    print [(k, v[0].data.shape) for k, v in classifier.params.items()]
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print classifier.blobs['data'].data[0]


    # conv1の出力
    #features = classifier.blobs['conv1'].data[0, :36]
    #plt.imshow(vis_square(features, padval=1))
    #plt.show()
     #conv2の出力
    #features = classifier.blobs['conv2'].data[0, :36]
    #vis_square(features, padval=1)
    #plt.imshow(vis_square(features, padval=1))
    #plt.show()
    # # conv3の出力
    #features = classifier.blobs['conv3'].data[0, :64]
    #vis_square(features, padval=1)
    #plt.imshow(vis_square(features, padval=1))
    #plt.show()
    # pool3の出力
    #features = classifier.blobs['pool3'].data[0, :64]
    #vis_square(features, padval=1)
    #plt.imshow(vis_square(features, padval=1))
    #plt.show()
    # 確率値の出力
    #print classifier.blobs['prob'].data




    # ウィンドウ破棄
    cv2.destroyAllWindows()
