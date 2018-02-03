
# face attractiveness prediction

## 1 Introduction

A toy program to predict attractiveness. 

### 1.1 facerank.py

| anaconda-version |  python2.7   |
| ---------- | ------ |
| boost      | 1.65.1 |
|   cmake         |   3.10.0     |
|    system-platform         |    win10    |

anaconda can be gotten from [official site ](https://www.anaconda.com/download/)
the detail on configuration of dlib [CSDN blog](http://blog.csdn.net/insanity666/article/details/72235275)

The program contains two part, training and prediction. There are some examples in code.

How the program runs？
* get the feature: img->68 face landmark->face attriveness feature(face ratio,more detail in[SCUT-FBP-paper](http://www.hcii-lab.net/data/SCUT-FBP/EN/introduce.html)) 
* get the model: we can use some traditional machine learning method to train and predict, like linear-regression, random forest.



### 1.2 facerank_xiaobin.py


| python-version | 3.5+ |
| ---------- | ---- |
| system-platform           |    win10  |

The program uses the api supported by microsoft xiaobin to predict attactiveness.

### 1.3 attention
 
To run the program examples, you should put some face pictures named 1.jpg 2.jpg 3.jpg and 4.jpg on the dir **data** and get detection model from [dlib official site](https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/download)

## 2 Datasource

To work on female face, I use the open data made by South China University of Technology, which can be get on [official site]( http://www.hciilab.net/data/SCUT-FBP/ )

The data also can be gotten from microsoft API of xiaobin attractiveness prediction.

## 3 Reference

The program  uses some code on [SCUT-FBP: A Benchmark Dataset for Facial Beauty Perception](http://www.hcii-lab.net/data/SCUT-FBP/EN/introduce.html) to get feature by face landmark.