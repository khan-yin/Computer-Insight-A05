import os
import torch
from src import face_detection
from src import report

def submit(imgdir, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    saveRoot = face_detection.SAVE_ROOT
    dirName = os.listdir(saveRoot)[-1]
    saveDir = os.path.join(saveRoot, dirName)
    # 此模型用于二分类人脸图像是否佩戴安全帽，输入人脸部分图像，输出戴安全帽的概率值
    model = face_detection.load_CNN_model(saveDir)
    useCuda = torch.cuda.is_available()
    rep = report.Report(useCuda)

    imgs = os.listdir(imgdir)
    for imgname in imgs:
        result = face_detection.detect_single_img(os.path.join(imgdir, imgname), model, rep, visualize=False)
        txtname = imgname.rstrip(".jpg") + ".txt"
        with open(os.path.join(savedir, txtname), "w") as f:
            f.writelines(result)




if __name__ == '__main__':
    submit("..\data\submit\JPEGImages", "..\data\submit\submit")