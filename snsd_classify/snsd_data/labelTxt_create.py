# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import os
import re
import random

# 学習データ用ラベル作成
# extension = .png, .jpg, etc
def create_train_label(extension, folderlist):
	path = "/home/akentu/caffe/examples/snsd_classify/"
	read_path = "snsd_data/train/"

	classify_folders = os.listdir(path + read_path)		# フォルダ内の全フォルダ・ファイルのリストを取得
	print 'In the "66_patch" dir, folder = %s' %classify_folders
	print 'Input Folder Name: folderlist = %s' %folderlist
	class_id = 0						# 分類ID
	for a in folderlist:
		# 対象フォルダを探す
		if a in classify_folders:
			# 対象フォルダ内の各画像フォルダ
			imagefolders = os.listdir(path + read_path + a)
			print 'In the %s dir, folder = %s' %(a, imagefolders)
			total = 0						# 合計枚数
			for b in imagefolders:
				# 各画像フォルダ内の指定拡張子のファイル数を調べる
				imagefolder_path = read_path + a + "/" + b	# 対象フォルダのパス

				image = os.listdir(path + imagefolder_path)    	# 画像を探すフォルダ
				count = 0					# カウンタの初期化
				for e in image:					# ファイルの数だけループ
			    		index = re.search(extension, e)		# 指定拡張子のものを検出
			    		if index:
						count = count + 1

				if count > 0:
					print("Images are %s in %s. Use half images." %(count, b))	# ファイル数の表示
					# 学習に使用する画像のパス＋ファイル名リストの作成
					filelist = []							# 使用対象画像のみのリスト
					if a == "non-road":
						image = random.sample(image, len(image)/2)
					for l in image:
					    	if os.path.isfile(path + imagefolder_path + "/" + l):	# 使用対象画像のパス
							filelist.append(imagefolder_path + "/" + l)
							total = total + 1
					print '%s images classified "%s" label.' %(a, class_id)

					# ファイル書き込み（無い場合は新規作成）
					train_txt = open('train.txt', 'a')				# 追記モード
					for i in filelist:
						train_txt.write(str(i) + " " + str(class_id) + "\n")	# 1行ずつ改行して書き込み
					train_txt.close()						# 必ず閉じる

				else:
					print("Designated files are not found in %s." %a)
			class_id = class_id + 1
			print "Total : %s" %total
		else:
			print("Designated ImageFolder is not found in 66_patch.")


# テストデータ用ラベル作成
# extension = .png, .jpg, etc
def create_test_label(extension, folderlist):
	path = "/home/akentu/caffe/examples/snsd_classify/"
	read_path = "snsd_data/test/"

	classify_folders = os.listdir(path + read_path)		# フォルダ内の全フォルダ・ファイルのリストを取得
	print 'In the "66_patch" dir, folder = %s' %classify_folders
	print 'Input Folder Name: folderlist = %s' %folderlist
	class_id = 0						# 分類ID
	for a in folderlist:
		# 対象フォルダを探す
		if a in classify_folders:
			# 対象フォルダ内の各画像フォルダ
			imagefolders = os.listdir(path + read_path + a)
			print 'In the %s dir, folder = %s' %(a, imagefolders)
			total = 0						# 合計枚数
			for b in imagefolders:
				# 各画像フォルダ内の指定拡張子のファイル数を調べる
				imagefolder_path = read_path + a + "/" + b	# 対象フォルダのパス
				image = os.listdir(path + imagefolder_path)	# 画像を探すフォルダ
				count = 0					# カウンタの初期化
				for e in image:					# ファイルの数だけループ
			    		index = re.search(extension, e)		# 指定拡張子のものを検出
			    		if index:
						count = count + 1

				if count > 0:
					print("Images are %s in %s. Use half images." %(count, b))	# ファイル数の表示
					# 学習に使用する画像のパス＋ファイル名リストの作成
					filelist = []							# 使用対象画像のみのリスト
#					if a == "etc":
#						image = random.sample(image, 148)
					for l in image:
					    	if os.path.isfile(path + imagefolder_path + "/" + l):	# 使用対象画像のパス
							filelist.append(imagefolder_path + "/" + l)
							total = total + 1
					print '%s images classified "%s" label.' %(a, class_id)

					# ファイル書き込み（無い場合は新規作成）
					test_txt = open('test.txt', 'a')				# 追記モード
					for i in filelist:
						test_txt.write(str(i) + " " + str(class_id) + "\n")	# 1行ずつ改行して書き込み
					test_txt.close()						# 必ず閉じる

				else:
					print("Designated files are not found in %s." %a)
			class_id = class_id + 1
			print "Total : %s" %total
		else:
			print("Designated ImageFolder is not found in 66_patch.")

# メイン文
def main():
	argv = sys.argv
	argc = len(argv)			# 引数の個数
	if (argc < 4):				# 引数が正しく入力されているかチェック
		print 'Usage: python %s ImageExtension　"train" or "test" ImageFolder_1 ImageFolder_2 ...' %argv[0]
		quit()
	
	extension = argv[1]	# 拡張子
	folderlist = []		# 対象フォルダのリスト作成
	for f in range(3, argc):
		folderlist.append(argv[f])

	# 学習用かテスト用か
	if argv[2] == "train":
		create_train_label(extension, folderlist)
	elif argv[2] == "test":
		create_test_label(extension, folderlist)
	else:
		print 'Usage: python %s ImageExtension　"train" or "test" ImageFolder_1 ImageFolder_2 ...' %argv[0]
		quit()
	print("Done")


if __name__ == '__main__':
	main()
