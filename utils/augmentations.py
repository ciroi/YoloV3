'''
Descripttion: 
version: 
Author: ZaoShan
Date: 2020-08-27 09:44:53
LastEditors: Zaoshan
LastEditTime: 2020-08-30 21:28:38
'''
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2 as cv
from PIL import Image
import math

# Hyperparameters
hyp = {'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

       
"mirror"
def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


''' cutmix '''
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# generate mixed sample
def cutmix(image1, target1, image2, target2, beta = 1.0):
    lam = np.random.beta(beta, beta)      # 1.0, we use uniform dist.
    # rand_index = torch.randperm(images.size()[0]).cuda()
    # target_a = targets
    # target_b = targets[rand_index]
    image_cutmix = image1.copy()
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_cutmix.size(), lam)
    image_cutmix[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image1.size()[-1] * image1.size()[-2]))
    target_cutmix = target1 * lam + target2 * (1. - lam)

    # compute output
    # output = model(images)
    # loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
    return image_cutmix, target_cutmix

''' gridmask '''
def gridmask(image, mode=1, rotate=1, r_ratio=0.5, d_ratio=1):
    _,h,w = image.size()

    # 1.5 * h, 1.5 * w works fine with the squared images
    # But with rectangular input, the mask might not be able to recover back to the input image shape
    # A square mask with edge length equal to the diagnoal of the input image 
    # will be able to cover all the image spot after the rotation. This is also the minimum square.
    L = math.ceil((math.sqrt(h*h + w*w)))

    d = np.random.randint(2, min(h,w)*d_ratio)
    # maybe use ceil? but i guess no big difference
    l = math.ceil(d*r_ratio)

    mask = np.ones((L, L), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(L//d):
            s = d*i + st_h
            t = s+l
            s = max(min(s, L), 0)
            t = max(min(t, L), 0)
            mask[s:t,:] *= 0
    for i in range(L//d):
            s = d*i + st_w
            t = s+l
            s = max(min(s, L), 0)
            t = max(min(t, L), 0)
            mask[:,s:t] *= 0
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(L-h)//2:(L-h)//2+h, (L-w)//2:(L-w)//2+w]

    mask = torch.from_numpy(mask).float() #cuda
    if mode == 1:
        mask = 1-mask

    mask = mask.expand_as(image)
    image = image * mask

    # image_ = image.clone().numpy()
    # image_ = np.transpose(image_, [1,2,0]) 
    # image__ = image_.copy()
    # image__[:,:,0] = image_[:,:,2]
    # image__[:,:,1] = image_[:,:,1]
    # image__[:,:,2] = image_[:,:,0]   #cv2读取的是bgr,转换成rgb就要做一下变通
    # cv.imshow("img0", image__)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return image


''' Mosaic: Promoted in Yolo v4 '''
# https://github.com/ultralytics/yolov3/blob/master/utils/datasets.py
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

def mosaic(images, targets, img_size):
    # loads images in a mosaic
    labels4 = []
    s = img_size
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    # 在可允许的范围之内，随机抽取4个indices （batchsize=16, [1, 0, 2, 4], [2, 0, 1, 4]）
    for i, img in enumerate(images):
        # Load image
        # img, _, (h, w) = load_image(self, index)
        (c, h, w) = img.size()
        img = img.numpy()
        img = np.transpose(img, [1,2,0])
        # img = img.copy() #shan
        # place img in img4
        if i == 0:  # top left
            # 把新图像先设置成原来的4倍，到时候再resize回去，114是gray
            img4 = np.full((s * 2, s * 2, c), 114 / 255,  dtype=np.float32)  # base image with 4 tiles 
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (new/large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (original/small image)
            # 回看ppt讲解
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b # 有时边上还是灰的
        padh = y1a - y1b

        # Labels
        # x = self.labels[index]
        x = targets[i]
        x = x.numpy()
        x = x.astype(np.float32)
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # for ii in range(x.shape[0]):
            #     xcc = x[ii, 1]
            #     ycc = x[ii, 2]
            #     w0 = x[ii, 3]
            #     h0 = x[ii, 4]
            #     x1 = int((xcc - w0 / 2)*w)
            #     y1 = int((ycc - w0 / 2)*h)
            #     x2 = int((xcc + w0 / 2)*w)
            #     y2 = int((ycc + w0 / 2)*h)
            #     img = cv.rectangle(img,(x1, y1),(x2, y2),(0,0,1),2) #画矩形框
        
            # 此时x是0-1，同时，label是[bbox_xc, bbox_yc, bbox_w, bbox_c]
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        
        labels4.append(labels)
        
        # cv.imshow("img0", img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    # Concat/clip labels
    if len(labels4):
        # a = np.array([[1, 2], [3, 4]])
        # c = np.concatenate(a, axis=0)
        # c: [1, 2, 3, 4]
        labels4 = np.concatenate(labels4, 0)    # 0是dimension
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

        lab4 = labels4.copy()
        x1 = lab4[:, 1]
        y1 = lab4[:, 2]
        x2 = lab4[:, 3]
        y2 = lab4[:, 4]
        labels4[:, 1] = ((x1 + x2) / 2) / (s * 2)        # newer center x
        labels4[:, 2] = ((y1 + y2) / 2) / (s * 2)        # newer center y
        labels4[:, 3] = (x2 - x1) / (s * 2)              # newer width
        labels4[:, 4] = (y2 - y1) / (s * 2)              # newer height
        index_d = np.where(labels4[:,3] <= (1.0 / (s * 2)))
        if index_d:
            labels4 = np.delete(labels4, index_d, axis=0)
        index_d = np.where(labels4[:,4] <= (1.0 / (s * 2)))
        if index_d:
            labels4 = np.delete(labels4, index_d, axis=0)

    # Augment
    # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
    # img4, labels4 = random_affine(img4, labels4,
    #                               degrees=hyp['degrees'],
    #                               translate=hyp['translate'],
    #                               scale=hyp['scale'],
    #                               shear=hyp['shear'],
    #                               border=(-s // 2, s // 2))  # border to remove

    # img4_ = img4.copy()
    # if labels4.size > 0:  # Normalized xywh to pixel xyxy format
    #     for ii in range(labels4.shape[0]):
    #         x1 = int((labels4[ii, 1] - labels4[ii, 3] / 2) * 2 *s)
    #         y1 = int((labels4[ii, 2] - labels4[ii, 4] / 2) * 2 *s)
    #         x2 = int((labels4[ii, 1] + labels4[ii, 3] / 2) * 2 *s)
    #         y2 = int((labels4[ii, 2] + labels4[ii, 4] / 2) * 2 *s)
    #         img4_ = cv.rectangle(img4_,(x1, y1),(x2, y2),(0,0,1),2) #画矩形框
    # img4__ = img4_.copy()
    # img4__[:,:,0] = img4_[:,:,2]
    # img4__[:,:,1] = img4_[:,:,1]
    # img4__[:,:,2] = img4_[:,:,0]   #cv2读取的是bgr,转换成rgb就要做一下变通
    # cv.imshow("img111", img4__)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    img4 = np.transpose(img4, [2,0,1]) 
    img4 = torch.from_numpy(img4) #转回tensor
    labels4 = torch.from_numpy(labels4)
    return img4, labels4