import os

import cv2
import numpy as np
import torch
from src import resnet
import datetime
from src import report

MODEL_ROOT = "../data/experiment"
OUTPUT_ROOT = "../data/output"
TEST_ROOT = "../data/test"
font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

def load_CNN_model(saveDir, useCuda=None):
    net = resnet.resnet(in_channel=3, num_classes=1)
    print("Loading model", saveDir)
    ckptPath = os.path.join(saveDir, "best_ckpt")
    if useCuda is None:
        useCuda = torch.cuda.is_available()

    if useCuda:
        ckpt = torch.load(ckptPath)
    else:
        ckpt =torch.load(ckptPath, map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])
    if useCuda:
        net.cuda()
    net.eval()
    return net


def detect_single_img(path, model=None, rep=None, visualize=True, scale=1,
                      modelRoot=None, dirId=-1, threshold=0.5, send_report = False):

    useCuda = torch.cuda.is_available()
    if model is None:
        if modelRoot is None:
            modelRoot = MODEL_ROOT
        dirName = os.listdir(modelRoot)[dirId]
        saveDir = os.path.join(modelRoot, dirName)
        # 此模型用于二分类人脸图像是否佩戴安全帽，输入人脸部分图像，输出戴安全帽的概率值
        model = load_CNN_model(saveDir)

    if rep is None:
        rep = report.Report(useCuda)

    img = cv2.imread(path)
    height, width = img.shape[:2]
    if scale != 1:
        img = cv2.resize(img, (int(scale*width), int(scale*height)), interpolation=cv2.INTER_CUBIC)
    # faces是检测到的人脸坐标列表
    faces, _ = rep.mtcnn_detector.detect_face(img)
    faces = rep.mtcnn_detector.box_expand(faces, 0.3, 0.2)

    if visualize:img_cp = img.copy()
    img_result = []
    # 遍历每一张图像中的人脸
    for bbox in faces:
        x, y = int(bbox[0]), int(bbox[1])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        # face是人脸区域缩放成96*96的正方形图像
        if w<=0 or h<=0 or x<0 or y<0: continue

        face = cv2.resize(img[y:y+h,x:x+w,:], (96, 96), interpolation=cv2.INTER_LINEAR)#rep.chopFace(img,(x, y, w, h), expand=False)
        img_tensor = rep.nparr2tensor(face)
        out = model(img_tensor)
        wear_hat_confidence = out.item()
        if wear_hat_confidence < threshold:
            img_result.append("person {:.3f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                1 - wear_hat_confidence, bbox[0], bbox[1], bbox[2], bbox[3]))
            if visualize:
                #如果戴安全帽的概率值小于threshold，触发报告，画红框
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                img = cv2.putText(img, "helmet: {:.2f}".format(wear_hat_confidence),
                                  (x-10, y-10), font, .8, (30, 30, 255), 2)
            if send_report:
                dat = datetime.datetime.now()
                cv2.rectangle(img_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
                rep.report(img_tensor, wear_hat_confidence, dat,
                           cv2.rectangle(img_cp, (x, y), (x + w, y + h), (0, 0, 255), 2))
        else:
            img_result.append("hat {:.3f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
                wear_hat_confidence, bbox[0], bbox[1], bbox[2], bbox[3]))
            if visualize:
                #画蓝框
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img = cv2.putText(img, "helmet: {:.2f}".format(wear_hat_confidence),
                                  (x-10, y+h+10), font, .8, (255, 30, 30), 2)

    if visualize:
        cv2.imshow("camera", img)
        cv2.waitKey(0)
    return img_result


def face_rec_camera(modelRoot=None, dirId=-1, scale=1, threshold=0.5):
    """
    此函数中是摄像头识别人脸，判别是否佩戴安全帽，存数据进数据库的主循环
    :param modelRoot: 
    :param dirId: 读取的卷积网络编号，-1表示最后一个，也就是训练阶段最后保存的
    :param threshold: 戴安全帽的概率值小于多少视为触发报告
    :return: 
    """
    if modelRoot is None:
        modelRoot = MODEL_ROOT
    dirName = os.listdir(modelRoot)[dirId]
    saveDir = os.path.join(modelRoot, dirName)

    useCuda = torch.cuda.is_available()
    # 此模型用于二分类人脸图像是否佩戴安全帽，输入人脸部分图像，输出戴安全帽的概率值
    model = load_CNN_model(saveDir)
    rep = report.Report(useCuda)

    print("Opening camera..")
    camera = cv2.VideoCapture(0)
    height, width = int(scale*camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(scale*camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        read, img = camera.read()
        if not read:
            break
        if scale != 1:
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        img_show = img.copy()
        # faces是检测到的人脸坐标列表
        faces, _ = rep.mtcnn_detector.detect_face(img)
        faces = rep.mtcnn_detector.box_expand(faces, 0.2, 0.2)
        # 遍历每一张图像中的人脸
        for bbox in faces:
            x, y = int(bbox[0]), int(bbox[1])
            w = int(bbox[2] - bbox[0])
            h = int(bbox[3] - bbox[1])
            if w<=0 or h<=0 or x<0 or y<0: continue
            # face是人脸区域缩放成96*96的正方形图像
            face = cv2.resize(img[y:y+h,x:x+w,:], (96, 96), interpolation=cv2.INTER_LINEAR)
            img_tensor = rep.nparr2tensor(face)
            out = model(img_tensor)
            wear_hat_confidence = out.item()
            if wear_hat_confidence < threshold:
                #如果戴安全帽的概率值小于threshold，触发报告，画红框
                img_show = cv2.rectangle(img_show, (x, y), (x+w, y+h), (0, 0, 255), 2)
                img_show = cv2.putText(img_show, "{:.2f}".format(wear_hat_confidence),
                                  (x-10, y-10), font, .8, (30, 30, 255), 2)

                img_report = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                img_report = cv2.putText(img_report, "{:.2f}".format(wear_hat_confidence),
                                  (x-10, y-10), font, .8, (30, 30, 255), 2)
                dat = datetime.datetime.now()
                rep.report(img_tensor, wear_hat_confidence, dat, img_report)
            else:
                #画蓝框
                img_show = cv2.rectangle(img_show, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img_show = cv2.putText(img_show, "{:.2f}".format(wear_hat_confidence),
                                  (x-10, y+h+20), font, .8, (255, 30, 30), 2)
        cv2.imshow("camera", img_show)
        if cv2.waitKey(1000 // 12) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

def face_rec_local(videoName, testRoot=None, outputRoot=None, modelRoot=None, saveOutput=True, scale=1, dirId=-1, threshold=0.5):
    """
    此函数中是摄像头识别人脸，判别是否佩戴安全帽，存数据进数据库的主循环
    :param modelRoot: 
    :param dirId: 读取的卷积网络编号，-1表示最后一个，也就是训练阶段最后保存的
    :param threshold: 戴安全帽的概率值小于多少视为触发报告
    :return: 
    """
    if testRoot is None:
        testRoot = TEST_ROOT
    if modelRoot is None:
        modelRoot = MODEL_ROOT
    if outputRoot is None:
        outputRoot = OUTPUT_ROOT
    dirName = os.listdir(modelRoot)[dirId]
    saveDir = os.path.join(modelRoot, dirName)

    useCuda = torch.cuda.is_available()
    # 此模型用于二分类人脸图像是否佩戴安全帽，输入人脸部分图像，输出戴安全帽的概率值
    model = load_CNN_model(saveDir, useCuda)
    rep = report.Report(useCuda)

    videoFile = os.path.join(testRoot, videoName)
    outputFile = os.path.join(outputRoot, "out.mp4")
    print("Reading video..")
    camera = cv2.VideoCapture(videoFile)
    if saveOutput:
        height, width = int(scale*camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(scale*camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sz = (height, width)
        fps = 30
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        vout_1 = cv2.VideoWriter()
        vout_1.open(outputFile,fourcc,fps,sz,True)

    while True:
        read, img = camera.read()
        if not read:
            break
        if scale != 1:
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)

        # faces是检测到的人脸坐标列表
        faces, _ = rep.mtcnn_detector.detect_face(img)
        faces = rep.mtcnn_detector.box_expand(faces, 0.2, 0.2)
        # 遍历每一张图像中的人脸
        for bbox in faces:
            x, y = int(bbox[0]), int(bbox[1])
            w = int(bbox[2] - bbox[0])
            h = int(bbox[3] - bbox[1])
            if w<=0 or h<=0 or x<0 or y<0: continue
            # face是人脸区域缩放成96*96的正方形图像
            face = cv2.resize(img[y:y+h,x:x+w,:], (96, 96), interpolation=cv2.INTER_LINEAR)
            img_tensor = rep.nparr2tensor(face)
            out = model(img_tensor)
            wear_hat_confidence = out.item()
            if wear_hat_confidence < threshold:
                #如果戴安全帽的概率值小于threshold，触发报告，画红框
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                img = cv2.putText(img, "{:.2f}".format(wear_hat_confidence),
                                  (x-10, y-10), font, .8, (30, 30, 255), 2)
                dat = datetime.datetime.now()
                # rep.report(img_tensor, wear_hat_confidence, dat, img_cp)
            else:
                #画蓝框
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img = cv2.putText(img, "{:.2f}".format(wear_hat_confidence),
                                  (x-10, y+h+20), font, .8, (255, 30, 30), 2)
        cv2.imshow("camera", img)
        vout_1.write(img)
        if cv2.waitKey(1000 // 12) & 0xff == ord('q'):
            break
    vout_1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_single_img("./img/001289.jpg", scale=1.2)

