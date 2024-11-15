import os

import numpy as np
import torch
from hausdorff import hausdorff_distance
from PIL import Image


def dice_coefficient(pred, gt, smooth=1e-5):
    """computational formula：
    dice = 2TP/(FP + 2TP + FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # if (pred.sum() + gt.sum()) == 0:
    #     return 1
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def sespiou_coefficient(pred, gt, smooth=1e-5):
    """computational formula:
    sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # pred_flat = pred.view(N, -1)
    # gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    return SE.sum() / N, SP.sum() / N, IOU.sum() / N


def sespiou_coefficient2(pred, gt, all=False, smooth=1e-5):
    """computational formula:
    sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # pred_flat = pred.view(N, -1)
    # gt_flat = gt.view(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth) / (TP + FP + FN + TN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2 * Precision * Recall / (Recall + Precision + smooth)
    if all:
        return (
            SE.sum() / N,
            SP.sum() / N,
            IOU.sum() / N,
            Acc.sum() / N,
            F1.sum() / N,
            Precision.sum() / N,
            Recall.sum() / N,
        )
    else:
        return IOU.sum() / N, Acc.sum() / N, SE.sum() / N, SP.sum() / N


def get_matrix(pred, gt, smooth=1e-5):
    """computational formula:
    sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    iou = TP/(FP+TP+FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    return TP, FP, TN, FN


def load_images(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(folder, filename)
            images[filename] = np.array(
                Image.open(path).convert("1").resize([512, 512])
            )  # 转换为灰度图
            images[filename]
    return images


def get_seg_metrics(GT_folder, Pre_folder):
    GT_images = load_images(GT_folder)
    Pre_images = load_images(Pre_folder)
    dice_scores = []
    hd95_scores = []
    iou_scores = []
    accuracy_scores = []
    sensitivity_scores = []

    for filename in Pre_images:

        pred = np.zeros_like(Pre_images[filename]).astype(np.float32)
        pred[Pre_images[filename] == 1] = 255
        gt = np.zeros_like(GT_images[filename]).astype(np.float32)
        gt[GT_images[filename] == 1] = 255

        pred = pred[None, :, :]
        gt = gt[None, :, :]

        dice_scores.append(dice_coefficient(pred, gt))
        hd95_scores.append(
            hausdorff_distance(pred[0, :, :], gt[0, :, :], distance="manhattan")
        )
        iou, acc, se, sp = sespiou_coefficient2(pred, gt, all=False)
        iou_scores.append(iou)
        accuracy_scores.append(acc)
        sensitivity_scores.append(se)

    # Calculate means and variances
    seg_metrics = {
        "Dice": (
            np.around(np.mean(dice_scores) * 100, decimals=3),
            np.around(np.std(dice_scores) * 100, decimals=3),
        ),
        "Hausdorff": (
            np.around(np.mean(hd95_scores), decimals=3),
            np.around(np.std(hd95_scores), decimals=3),
        ),
        "IoU": (
            np.around(np.mean(iou_scores) * 100, decimals=3),
            np.around(np.std(iou_scores) * 100, decimals=3),
        ),
        "Accuracy": (
            np.around(np.mean(accuracy_scores) * 100, decimals=3),
            np.around(np.std(accuracy_scores) * 100, decimals=3),
        ),
        "Sensitivity": (
            np.around(np.mean(sensitivity_scores) * 100, decimals=3),
            np.around(np.std(sensitivity_scores) * 100, decimals=3),
        ),
    }

    return seg_metrics


def get_seg_fromarray(GT_array, Pre_array):
    device = GT_array.device
    GT_array = GT_array * 255
    Pre_array = Pre_array * 255

    B, H, W = GT_array.shape
    dice_scores = torch.zeros(B).to(device)
    hd95_scores = torch.zeros(B).to(device)
    iou_scores = torch.zeros(B).to(device)
    accuracy_scores = torch.zeros(B).to(device)
    sensitivity_scores = torch.zeros(B).to(device)
    specificity_scores = torch.zeros(B).to(device)
    for i in range(B):
        pred = Pre_array[i : i + 1, :, :]
        gt = GT_array[i : i + 1, :, :]

        dice_scores[i] = dice_coefficient(pred, gt)
        hd95_scores[i] = torch.tensor(
            hausdorff_distance(
                pred[0, :, :].cpu().numpy(),
                gt[0, :, :].cpu().numpy(),
                distance="manhattan",
            )
        ).to(device)
        iou, acc, se, sp = sespiou_coefficient2(pred, gt, all=False)
        iou_scores[i] = iou
        accuracy_scores[i] = acc
        sensitivity_scores[i] = se
        specificity_scores[i] = sp

    # Calculate means and variances
    seg_metrics = {
        "Dice": (torch.mean(dice_scores), torch.std(dice_scores)),
        "Hausdorff": (torch.mean(hd95_scores), torch.std(hd95_scores)),
        "IoU": (torch.mean(iou_scores), torch.std(iou_scores)),
        "Accuracy": (torch.mean(accuracy_scores), torch.std(accuracy_scores)),
        "Sensitivity": (torch.mean(sensitivity_scores), torch.std(sensitivity_scores)),
        "Specificity": (torch.mean(specificity_scores), torch.std(specificity_scores)),
    }

    return seg_metrics


if __name__ == "__main__":
    # 文件夹路径
    GT_folder = None
    Pre_folder = None
    seg_metrics = get_seg_metrics(GT_folder, Pre_folder)
    print(seg_metrics)
