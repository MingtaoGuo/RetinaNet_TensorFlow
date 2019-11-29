# RetinaNet_TensorFlow
Implementation of RetinaNet by TensorFlow (object detection)

# Introduction
![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/introduction.jpg)

### Focal Loss
![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/formula.jpg)
``` python
def focal_loss(logits, labels, alpha=0.25, gamma=2):
    pos_pt = tf.clip_by_value(tf.nn.sigmoid(logits), 1e-10, 0.999)
    fl = labels * tf.log(pos_pt) * tf.pow(1 - pos_pt, gamma) * alpha + (1 - labels) * tf.log(1 - pos_pt) * tf.pow(pos_pt, gamma) * (1 - alpha)
    fl = -tf.reduce_sum(fl, axis=2)
    return fl
```
# How to use
### Dataset
Pascal Voc: http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
### Training phase
1. Downloading the pre-trained model of ResNet50, and put it into the folder **resnet_ckpt** 
   
   Address: http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz

2. According to your own dataset, modify the **config.py** by yourself
3. Executing **train.py** 
### Testing phase
1. Changing the IMG_PATH or VIDEO_PATH in **test.py** for testing
2. Executing **test.py**

    Model we trained: https://drive.google.com/open?id=1_j-bjQ_SWT3txqCabM8_Ny0IMUqTb8Lk
# Requirement
1. python
2. tensorflow
3. pillow
4. numpy
5. cv2
# Results
|Total Loss|Class Loss|Box Loss|
|-|-|-|
|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/total_loss.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/class_loss.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/box_loss.jpg)|

||||
|-|-|-|
|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/1.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/8.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/6.jpg)|
|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/7.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/9.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/4.jpg)|
|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/2.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/3.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/14.jpg)|
|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/10.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/12.jpg)|![](https://github.com/MingtaoGuo/RetinaNet_TensorFlow/blob/master/IMGS/13.jpg)|

# Reference
[1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.
# Author
Mingtao Guo
