# -*- coding: utf-8 -*-
import glob
import urllib.request
import base64
import time
import random
import json
import re
import pickle
import threading
import time
from urllib.parse import urlencode

import os

upload_url = 'http://kan.msxiaobing.com/Api/Image/UploadBase64'
comp_url = 'https://kan.msxiaobing.com/Api/ImageAnalyze/Process?service=yanzhi&tid=079f15a4cb814b8ba509b633790eebfb'

faceranks = {}
THREADING_NUMBER=threading.Semaphore(10)
current_Number=10
lock=threading.Lock()


def __get_img_url(file_path):
    """
    将图片上传到microsoft服务器，等待颜值验证。函数返回microsoft服务器上传图片的地址
    :param file_path: 待上传的文件地址
    :return:microsoft图片服务器中我方上传图片的位置
    """
    with open(file_path,'rb') as f:
        img_base64=base64.b64encode(f.read())
    req=urllib.request.Request(upload_url,data=img_base64)
    data = urllib.request.urlopen(req).read()
    data=str(data, encoding='utf-8')
    responseInfo=json.loads(data)
    url=responseInfo['Host'] + responseInfo['Url']
    return url


def __get_response(img_url):
    """

    :param img_url: microsoft oss 中上传图片的位置
    :return: microsoft 颜值评估的返回
    """
    sys_time = int(time.time())
    data = {
        'MsgId': str(sys_time) + '733',
        'CreateTime': sys_time,
        'Content[imageUrl]': img_url,
    }
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
        'Cookie':'_ga=GA1.2.1597838376.1504599720; _gid=GA1.2.1466467655.1504599720; ai_user=sp1jt|2017-09-05T10:53:04.090Z; cpid=YDLcMF5LPDFfSlQyfUkvMs9IbjQZMiQ2XTJHMVswUTFPAA; salt=EAA803807C2E9ECD7D786D3FA9516786; ARRAffinity=3dc0ec2b3434a920266e7d4652ca9f67c3f662b5a675f83cf7467278ef043663; ai_session=sQna0|1504664570638.64|1504664570638'+str(random.randint(11, 999)),
        'Referer': 'https://kan.msxiaobing.com/ImageGame/Portal?task=yanzhi&feid=d89e6ce730dab7a2410c6dad803b5986',

    }
    data = urlencode(data).encode('utf-8')
    req = urllib.request.Request(comp_url,headers=headers,data=data)
    response=urllib.request.urlopen(req).read()
    response=str(response, encoding='utf-8')
    responseInfo = json.loads(response)

    return responseInfo

def __parse_score(responsedata):
    pattern = "\d\.\d"
    score=0
    if re.findall(pattern,responsedata["content"]['text']):
        score=re.findall(pattern,responsedata["content"]['text'])[0]
        return score
    else:
        return None


def get_facescore_fromimg(imgpath):
    """
    :param imgpath:需要被检测颜值的图片
    :return:None 如果检测失败。如果检测成功，返回一个浮点数分数，在0到10之间
    """
    url = __get_img_url(imgpath)
    responseinfo = __get_response(url)
    score = __parse_score(responseinfo)

    return score

def add_facescore_tofaceranks(imgpath):
    score=get_facescore_fromimg(imgpath)
    if score==None:
        os.remove(imgpath)
    else:

        if "/" in imgpath:
            imgpath = imgpath.split("/")[-1]
        if "\\" in imgpath:
            imgpath = imgpath.split("\\")[-1]
        imgname = imgpath.split(".")[0]
        lock.acquire()
        faceranks[imgname] = score
        lock.release()


    THREADING_NUMBER.release()
    lock.acquire()
    global current_Number
    current_Number += 1
    lock.release()

def get_facescores_fromimgdir(dirpath):
    """
    多线程获取图片的颜值，并且会将获取的图片从目录中删除
    起到分发线程的作用，返回的结果在全局变量faceranks中
    :param dirpath: 需要被批量检测颜值的图片所在目录

    """
    for filepath in glob.glob(os.path.join(dirpath,"*.jpg")):
        THREADING_NUMBER.acquire()
        lock.acquire()
        global current_Number
        current_Number-=1
        lock.release()


        t = threading.Thread(target=add_facescore_tofaceranks, args=(filepath,))
        t.start()

        time.sleep(1.5)


    #save the faceratings
    while 1:
        if current_Number==10:
            file = open("./model/faceratings.pkl", 'wb')
            pickle.dump(faceranks, file)
            print("faceratings done")
            break


if __name__=="__main__":
    #这是一个获取单个图片颜值的实例
    score=get_facescore_fromimg("./data/4.jpg")
    print(score)
    #这是一个获取目录中所有图片颜值的实例
    get_facescores_fromimgdir("./data")
    for k, v in faceranks.items():
        print(k + " " + v)
