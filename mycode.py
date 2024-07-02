import os
import json
import numpy as np
import cv2



#このコードの目的は、指定されたclass_fileからクラスの情報を読み取り、classesという変数にリストとして格納することです。その後の処理で、このリストを使用してCOCO形式のデータを作成します。
def yolo_to_coco(yolo_path, coco_path, class_file):
    # クラス（ラベル）ファイルの読み込み
    with open(class_file, 'r') as f:            #f.read()は、ファイルの内容を一括で読み取ります。
        classes = f.read().splitlines()         #ラベルの各行を取得？　ラベルの数だけclasesが存在する　splitlines()は、読み取ったファイルの内容を改行文字（\n）で分割し、リストとして返します。これにより、クラス（ラベル）ファイル内の各行がリストの要素になります。結果として、classesという変数には、クラスファイル内の各行が含まれるリストが格納されます。

    # YOLO形式のアノテーションデータを読み込みながらCOCO形式に変換
    coco_data = {       #coco_dataという変数は、COCO形式のデータを格納するための辞書です。この辞書は、'info'、'licenses'、'categories'、'images'、'annotations'というキーを持ちます。
        'info': {},
        'licenses': [],
        'categories': [{'id': i+0, 'name': classes[i], 'supercategory': 'object'} for i in range(len(classes))],                
        'images': [],                                                                           #len(classes)はクラスの総数,
        'annotations': []
        

    }


    image_id = 0
    annotation_id = 0

    for filename in os.listdir(yolo_path):      #指定されたディレクトリ（yolo_path）内のすべてのファイルをループで処理し、
        if filename.endswith('.txt'):           #拡張子が.txtで終わるファイルを選択
            # 画像ファイルのパスを取得
            image_filename = os.path.splitext(filename)[0] + '.jpg' #同じ名前だから.os.path.splitext　で.txtの拡張子を除いて.jpgつける          os.path.splitext(filename)[0]は、filenameの拡張子を除いた部分を取得しています。例えば、filenameがimage001.txtの場合、os.path.splitext(filename)[0]はimage001となります。
            image_path = os.path.join(yolo_path, image_filename)    #yolo_pathとimage_filenameを結合して、画像ファイルの絶対パスを作成しています

            # 画像の情報を取得
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # 画像情報をCOCOデータに追加
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_filename,
                'width': width,
                'height': height
            })

            # アノテーションデータの読み込み
            with open(os.path.join(yolo_path, filename), 'r') as f:
                lines = f.readlines()       #行ごとに読んでリスト化

            for line in lines:  #lineを使用して各行の内容にアクセスできる　各行に対してループ   bboxが2つだったら、2行読み込んだ理しないといけないから
                line = line.strip().split() #文字列の前後の空白文字を取り除いた後にスペースで分割する操作を行う    yoloデータが空白で開いてるから、それで分けて配列に入れてく感じ ["1", "0.511407", "0.522843", "0.186312", "0.335025"]

                class_id = int(line[0])     ########################################
                x_center = float(line[1])
                y_center = float(line[2])
                bbox_width = float(line[3])
                bbox_height = float(line[4])

                x_min = float((x_center - bbox_width/2) * width)
                y_min = float((y_center - bbox_height/2) * height)
                bbox_width = float(bbox_width * width)
                bbox_height = float(bbox_height * height)

                # アノテーション情報をCOCOデータに追加
                coco_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': class_id + 0,  # クラスIDに0を加える
                    'bbox': [x_min, y_min, bbox_width, bbox_height],
                    'area': bbox_width * bbox_height,
                    'iscrowd': 0
                })

                annotation_id += 1

            image_id += 1

    # COCOデータをJSONファイルに保存
    with open(coco_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


# 使用例
#yolo_to_coco('/path/to/yolo_annotations', '/path/to/coco_annotations.json', '/path/to/class_file.txt')
yolo_to_coco('/Users/shibata/Desktop/hinoTopPy/validation', '/Users/shibata/Desktop/hinoTop/validation/labels.json', '/Users/shibata/Desktop/Got/yolo2yolo2coco/class_file.txt')
