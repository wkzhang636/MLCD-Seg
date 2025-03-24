'''

This example focuses on how to use MLCD-Seg to perform inference and save the inference results

@DeepGlint 2025

'''



from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image


model_path = "DeepGlint-AI/MLCD-Seg" # or use your local path
mlcd_seg = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# Assuming you have an image named test.jpg
seg_img = Image.open("asserts/example.jpg").convert('RGB')
seg_prompt = "Could you provide a segmentation mask for the right giraffe in this image?"
pred_mask = mlcd_seg.seg(seg_img, seg_prompt, tokenizer, force_seg=True)
if pred_mask.shape[0] == 0:
    pred_mask = torch.zeros([1, *pred_mask.shape[1:]]).int()
mask_image = Image.fromarray(pred_mask.to(device="cpu").numpy()[0] * 255)
mask_image = mask_image.convert("1")
mask_image.save("output_mask.jpg") # or use mask_image.open() to show
