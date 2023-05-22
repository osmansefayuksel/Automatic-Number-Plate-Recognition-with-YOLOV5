# Automatic Number Plate Recognition with YOLOV5

## Welcome to automatic Number Plate recognition project

Firstly, YOLOV5 (You Only Look Once) algorithm was used in this project. We create our model by training our data in the Yolov5 algorithm. Since the labels in our data are in XML format, we need to make them suitable for the YOLOv5 algorithm. We have a pre-processing for this. We are making it ready for training in YOLOv5 algorithm. After completing the training, we export our model. 

## Train - Val Results
<br>

![Train_batch](https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/train_batch0.jpg)

<br>
<br>

![Train_batch](https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/train_batch2.jpg)

<br>
<br>

Then we determine the location of the plates in input image with the necessary functions. 

<br>
<br>

## val_batch 

<br>

![Val_batch](https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/val_batch0_labels.jpg)


## DETECTION RESULTS

![Detection 1](https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/result3.png)


<br>
<br>


![Detection 2](https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/result4.png)


<br>
<br>


After the plate detection, the OCR algorithm is pre-processed by crop the plate part so that the plate can read the letters and numbers more easily. The pre-processing process is very important for OCR

<br>
<br>

## CROPPED PLATE RESULTS
<br>

![Cropped 1]("https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/cropped3.png")


<br>
<br>


![Cropped 2](https://github.com/osmansefayuksel/Automatic-Number-Plate-Recognition-with-YOLOV5/blob/main/results/cropped4.png)


<br>
<br>


## [YOLOV5](https://github.com/ultralytics/yolov5)
<br>

## [For Dataset](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection)

## [Resources](https://www.kaggle.com/code/prateekcse101/mini-project-i)







