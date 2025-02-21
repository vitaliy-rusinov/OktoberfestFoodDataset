# Oktoberfest Food Dataset
The data was aquired during Schanzer Almfest at Ingolstadt in 2018 by [IlassAG](https://www.ilass.com). As part of a practical at the [Data Mining and Analytics Chair by Prof. Günnemann at TUM](https://www.kdd.in.tum.de) we were given the task to count objects at checkout. Therefore we annotated the data with bounding boxes and classes to train an object detection network.
![Annotated image](images/example_annotated.png)


## Download

You can find the dataset [here](https://mediatum.ub.tum.de/1487154)
- `dataset` contains the train and test datasets including the labels
  - the labels can be found in `files.txt` (OpenCV style)
  - `<filename> <number of objects> <classid1> <x1> <y1> <w1> <h1> <classid2> <x2> <y2> <w2> <h2> ...`
- `models` contains our pretrained tensorflow models (see [Preview.ipynb](Preview.ipynb) for an example usage)
- `video_data_zipped` contains the raw videos from which the dataset were extracted

The dataset in the PASCAL_VOC format can be found here

https://drive.google.com/open?id=1rgJUEFB4Cmbf9mQVdGPCHGiT4bvh_gDT (images)

https://drive.google.com/open?id=1mLIc1Ybs1rVwzMDuWMwxWUgl7spx2tBB (video)

In addition, the labels in the PASCAL_VOC format are available in the PASCAL_VOC folder.

Online Notebooks to train Faster RCNN and Retinanet models on the dataset using Google Colaboratory are available here

Faster RCNN Pytorch

https://drive.google.com/open?id=1CDQ5cIA8qsdm-OinbfPKM5DuoI6ewvZH

RetinaNet Tensorflow

https://drive.google.com/open?id=1KxP-j0TSQ_PY7xkJ4JNRyMnLv7kjRB_e


## Dataset Description


### Data Distribution

Class Id | Class | Images | Annotations | average quantity
 --- | --- | --- | --- | ---
0 | Bier | 300 | 436 | 1.45 
1 | Bier Mass | 200 | 299 | 1.50 
2 | Weissbier | 229 | 298 | 1.30 
3 | Cola | 165 | 210 | 1.27 
4 | Wasser | 198 | 284 | 1.43 
5 | Curry-Wurst | 120 | 159 | 1.32 
6 | Weisswein | 81 | 105 | 1.30 
7 | A-Schorle | 90 | 98 | 1.09 
8 | Jaegermeister | 43 | 152 | 3.53 
9 | Pommes | 110 | 126 | 1.15 
10 | Burger | 105 | 122 | 1.16 
11 | Williamsbirne | 50 | 121 | 2.42 
12 | Alm-Breze | 100 | 114 | 1.14 
13 | Brotzeitkorb | 65 | 72 | 1.11 
14 | Kaesespaetzle | 92 | 100 | 1.09 
  || Total | 1110 | 2696 | 2.43

### Statistics

![Images per class](images/images_per_class.png) \
![Annotations per class](images/annotations_per_class.png) \
![Items per image](images/items_per_image.png) \
![Occurance heat map](images/Occurance_heatmap.png)

## Benchmark
In order to train object detection models, we used [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and standalone Pytorch models. We trained several different models and got the best results for the Single Shot Detector with Feature Pyramid Networks (RetinaNet). Our evaluation metric was the area under the precision-recall curve (AUC) on a test set of 85 images (we ignored localization as our goal was counting objects).  We also used the data from the last two days to evaluate the models trained on the data from the first 9 days.

Approach | Backbone model | AUC | Example precision@recall
 --- | --- | --- | --- 
[SSD](https://dataserv.ub.tum.de/index.php/s/m1487154/download?path=/models&files=ssd.pb) | Mobilenet | 0.86 / - | 0.85@0.70 
[SSD + FPN](https://dataserv.ub.tum.de/index.php/s/m1487154/download?path=/models&files=ssd_fpn.pb) | Mobilenet | 0.98/0.92 | 0.97@0.97 (0.88@0.90)
[RFCN](https://dataserv.ub.tum.de/index.php/s/m1487154/download?path=/models&files=rfcn.pb) | ResNet-101 | 0.97/0.89 | 0.90@0.95 (0.72@0.90)
Faster RCNN | VGG-16 | 0.98/0.93 | 0.97@0.90 (0.79@0.90)

The numbers on the left denote the result on the small test set, the ones on the right - on the test set of data for the last two days (819 images
for training, 222 images for testing).

## Code
The [Evaluation](evaluation) folder contains Jupyter notebooks to evaluate the TensorFlow models.

With the [Preview](Preview.ipynb) notebook one can try out the pretrained TensorFlow models on arbitrary images.

The [CreateTFRecordFile](CreateTFRecordFile.ipynb) notebook contains code to convert the dataset in to the TFRecord file format so it can be used with the TensorFlow object detection library.

The [ShowAnnotations](ShowAnnotations.py) visualizes the bounding boxes of the dataset. Use 'n' for the next image, 'p' for the previous and 'q' to quit. 

## Authors
[Vitalii Rusinov](https://github.com/vitaliy-rusinov): Student of Informatics (M.Sc.) at TUM \
[Alexander Ziller](https://github.com/a1302z): Student of Robotics, Cognition & Intelligence (M.Sc.) at TUM \
[Julius Hansjakob](https://github.com/polarbart): Student of Informatics (M.Sc.) at TUM 

We also want to credit [Daniel Zügner](https://github.com/danielzuegner) for advising us any time and encouraging to publish this dataset. 
