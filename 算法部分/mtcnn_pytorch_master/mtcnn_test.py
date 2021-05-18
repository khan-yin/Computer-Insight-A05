import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    use_cuda = True
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./img/part2_002268.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    print(bboxs)
    # print box_align
    save_file = './img/result.jpg'
    bboxs = mtcnn_detector.box_expand(bboxs, 0.3, 0.25)
    vis_face(img_bg, bboxs, landmarks, save_file)
