#coding:utf-8
from ultralytics import YOLO
import cv2
from PIL import ImageFont
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

def get_license_result(ocr, image):
    license_nameAll = ""
    results = ocr.ocr(image, cls=True)[0]
   
    for result in results:
      if result:
          license_name, conf = result[1]
          # print("------result[0][1]-------" + license_name)
          if '·' in license_name:
              license_name = license_name.replace('·', '')
          license_nameAll = license_nameAll + " " +  license_name
      
    return license_nameAll, conf

# picture address
img_path = "/content/ultralytics/images/1.jpg"
now_img = cv2.imread(img_path)

# ocr model
#cls_model_dir = 'paddleModels/whl/cls/ch_ppocr_mobile_v2.0_cls_infer'
#rec_model_dir = 'paddleModels/whl/rec/ch/ch_PP-OCRv4_rec_infer'
# ocr = PaddleOCR(use_angle_cls=False, lang="ch", det=False)
ocr = PaddleOCR(lang="ch", det=False, rec_model_dir=rec_model_dir)

# yolov8 model
path = '/content/ultralytics/runs/detect/train7/weights/best.pt'

# conf  0.25    object confidence threshold for detection
# iou   0.7 int.ersection over union (IoU) threshold for NMS
model = YOLO(path, task='detect')
# model = YOLO(path, task='detect',conf=0.5)
results = model(img_path)[0]

location_list = results.boxes.xyxy.tolist()

if len(location_list) >= 1:
    location_list = [list(map(int, e)) for e in location_list]
    # get the location of license
    license_imgs = []
    for each in location_list:
        x1, y1, x2, y2 = each
        cropImg = now_img[y1:y2, x1:x2]
        license_imgs.append(cropImg)
        cv2.imwrite('carplate.jpg', cropImg)
        #cv2.imshow('111',cropImg)
        #cv2.waitKey(0)
    
    # the result of yolo 车牌识别结果
    lisence_res = []
    conf_list = []
    for each in license_imgs:
        license_num, conf = get_license_result(ocr, each)
        #print("-----pic--license_num------->" + license_num)
        if license_num:
            lisence_res.append(license_num)
            conf_list.append(conf)
        else:
            lisence_res.append('unrecognized')
            conf_list.append(0)
    for text, box in zip(lisence_res, location_list):
        cv2.rectangle(now_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        font_scale = 1.0  # 字体缩放比例
        font_thickness = 2  # 字体线条宽度
        text_position = (box[0], box[1] - 10)  # 假设文本位置在矩形框上方10个像素
        #print("-----pic--text------->" + text)
        cv2.putText(now_img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 255, 0),font_thickness)


now_img = cv2.resize(now_img,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
# save image
cv2.imwrite('output.jpg', now_img)

#cv2.imshow("YOLOv8 Detection", now_img)
#cv2.waitKey(0)