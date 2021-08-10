import argparse

from matplotlib import patches

from yolov5.utils.datasets import LoadImages
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import torch
import numpy as np
from PIL import Image
from yolov5.utils.torch_utils import select_device


def plot_image(image, boxes):
    """画出预测框"""
    im = np.array(image)

    height, width, _ = im.shape
    print(height,width)
    height = 1
    width = 1
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # Create a Rectangle potch
    for box in boxes:
        box = box[:4]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (box[0], box[1]),
            (box[2] - box[0]),
            (box[3] - box[1]),
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()

def detect(opt):
    device = select_device(opt.device)
    half = device.type != 'cpu'

    device = select_device(opt.device)
    model = attempt_load(opt.yolo_weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    if half:
        model.half()

    # img = np.array(Image.open('yolov5/data/images/bus.jpg'))
    # img = torch.from_numpy(img)

    dataset = LoadImages("yolov5/data/images/", img_size=imgsz)

    for (path, img, im0s, vid_cap) in dataset:
        img = torch.from_numpy(img)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        pred = pred[0]
        # pred = pred[...,0:-2]
        print("\n---",pred)
        print(img.shape)

        plot_image(img[0].permute(1, 2, 0).to("cpu"), pred)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5x.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                        help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)