from transformers import AutoModel, AutoTokenizer
from PIL import Image
from pathlib import Path
from pycocotools import mask as mask_util
from tqdm import tqdm
import cv2
import torch
import json
import numpy as np
import shortuuid
import argparse
import math
import os


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output:{output.shape }, target:{target.shape}"
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval(chunk_idx: int, all_device, coco_base: str, refcoco_json: str):
    if not isinstance(coco_base, Path):
        coco_base = Path(coco_base)
    model = AutoModel.from_pretrained(
        "DeepGlint-AI/MLCD-Seg",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained("DeepGlint-AI/MLCD-Seg", use_fast=False)
    with open(refcoco_json) as question_file:
        questions = json.loads(question_file.read())
    sub_questions = get_chunk(questions, all_device, chunk_idx)
    ans_file = open("test/ans_file_chunk_{}.json".format(chunk_idx), "w")
    for line in tqdm(sub_questions):
        try:
            idx = line["sample_id"]
        except:
            idx = line["id"]
        question_type = line["metadata"]["question_type"]
        dataset_name = line["metadata"]["dataset"]
        split_name = line["metadata"]["split"]
        image_file = line["image"]
        convs = line["conversations"]
        real_image = Image.open(coco_base / image_file).convert('RGB')
        collect_size = list(set([np.array(real_image).shape[:2]]))
        if len(collect_size) == 0:
            mask_h, mask_w = 336, 336
        elif len(collect_size) == 1:
            mask_h, mask_w = collect_size[0]
        else:
            areas = [h*w for (h, w) in collect_size]
            mask_h, mask_w = collect_size[areas.index(max(areas))]
        pred_mask = []
        for index, conv in enumerate(convs):
            if conv["from"] != "human":
                continue
            pred_mask.append(model.seg(real_image, conv["value"].replace("The <image> provides an overview of the picture.\n", ""), tokenizer, force_seg=True))
        pred_mask = torch.cat(pred_mask, dim=0)
        for index, image in enumerate(pred_mask):
            mask_image = Image.fromarray(image.cpu().numpy() * 255)
            mask_image = mask_image.convert('1')
            mask_image.save(f"seg_{index}.jpg")
        masks_list = []
        for img_i in range(1):
            if "segmentation" in line:
                masks = []
                for rle in line["segmentation"][img_i]:
                    m = mask_util.decode(rle)
                    m = cv2.resize(m, (mask_w, mask_h)).astype(np.uint8)
                    masks.append(m)
                masks = np.stack(masks, axis=0)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(0, mask_h, mask_w, dtype=torch.uint8)
            masks_list.append(masks)
        masks_list = torch.cat(masks_list, dim=0).float()
        masks_list = (masks_list > 0).int()
        intersection, union, accuracy_iou = 0.0, 0.0, 0.0
        for target, prediction in zip(masks_list, pred_mask):
            target = target.to(prediction.device)
            intersect, union_, _ = intersectionAndUnionGPU(
                prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
            )
            intersection += intersect
            union += union_
            accuracy_iou += intersect / (union_ + 1e-5)
            # handles no-object targets
            accuracy_iou[union_ == 0] += 1.0
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        accuracy_iou = accuracy_iou.cpu().numpy() / masks_list.shape[0]
        
        intersection = [float(intersection[0]), float(intersection[1])]
        union = [float(union[0]), float(union[1])]
        accuracy_iou = [float(accuracy_iou[0]), float(accuracy_iou[1])]

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "dataset": dataset_name,
                                   "split_name":split_name,
                                   "sample_id": idx,
                                   "intersection":intersection,
                                   "accuracy_iou":accuracy_iou,
                                   "union":union,
                                   "shortuuid": ans_id,
                                   "model_id": "mlcd_seg",
                                   "question_type": question_type,
                                   }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--all-device", type=int, default=8)
    parser.add_argument("--coco-base", type=str, default="./coco_base")
    parser.add_argument("--refcoco-json", type=str, default="./refcoco.jsonl")
    args = parser.parse_args()
    eval(args.device_id, args.all_device, args.coco_base, args.refcoco_json)