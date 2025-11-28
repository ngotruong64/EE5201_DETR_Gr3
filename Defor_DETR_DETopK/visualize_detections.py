# visualize_detections.py
import argparse
import os
import random

import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

from main import get_args_parser as get_main_args_parser
from models import build_model
from datasets import build_dataset
from util.misc import nested_tensor_from_tensor_list


def get_vis_args_parser():
    parser = argparse.ArgumentParser("Visualize Deformable DETR detections")
    parser.add_argument("--resume", type=str, required=True,
                        help="path to checkpoint, vd: output/checkpoint0049.pth")
    parser.add_argument("--split", type=str, default="val",
                        help="dataset split: train / val / test (tùy repo, thường là val)")
    parser.add_argument("--num_images", type=int, default=5,
                        help="số ảnh muốn vẽ")
    parser.add_argument("--score_thresh", type=float, default=0.5,
                        help="ngưỡng score để vẽ bbox")
    parser.add_argument("--save_dir", type=str, default="vis_results",
                        help="folder lưu ảnh kết quả")
    return parser


@torch.no_grad()
def run_visualization():
    # tách args dùng cho visualize và args dùng cho main.py
    vis_args, remaining = get_vis_args_parser().parse_known_args()
    main_args = get_main_args_parser().parse_args(remaining)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build dataset (val/test)
    dataset = build_dataset(vis_args.split, main_args)

    # 2. Build model
    model, _, postprocessors = build_model(main_args)
    model.to(device)
    model.eval()

    # 3. Load checkpoint
    assert os.path.exists(vis_args.resume), f"Không tìm thấy checkpoint: {vis_args.resume}"
    ckpt = torch.load(vis_args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from {vis_args.resume}")

    # 4. Chuẩn bị folder lưu
    os.makedirs(vis_args.save_dir, exist_ok=True)

    # 5. Lấy random index ảnh
    num_imgs = min(vis_args.num_images, len(dataset))
    indices = random.sample(range(len(dataset)), num_imgs)

    # ImageNet mean/std (thường dataset dùng cái này)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    for i, idx in enumerate(indices):
        sample, target = dataset[idx]      # sample: Tensor [3,H,W], đã normalize
        orig_size = target["orig_size"]    # (H, W) trong nhiều repo

        # 5.1 Chuẩn bị input cho model
        samples = nested_tensor_from_tensor_list([sample.to(device)])
        outputs = model(samples)

        # 5.2 Post-process về box gốc ảnh
        orig_h, orig_w = orig_size.tolist()
        orig_sizes = torch.as_tensor([[orig_h, orig_w]], device=device)
        results = postprocessors["bbox"](outputs, orig_sizes)[0]

        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()  # xyxy trong toạ độ ảnh gốc

        # 5.3 Lọc theo score
        keep = scores >= vis_args.score_thresh
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # 5.4 Unnormalize ảnh về dạng xem được
        img = sample.clone()
        for c in range(3):
            img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
        img = img.clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 5.5 Vẽ bbox
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # lấy tên class từ coco nếu có
            if hasattr(dataset, "coco"):
                cat_id = dataset.coco.getCatIds()[label]
                class_name = dataset.coco.cats[cat_id]["name"]
            else:
                class_name = f"class_{label}"

            text = f"{class_name} {score:.2f}"
            cv2.putText(img_np, text, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 5.6 Lưu & show
        out_path = os.path.join(vis_args.save_dir, f"vis_{i:03d}_idx{idx}.jpg")
        cv2.imwrite(out_path, img_np)
        print(f"Saved {out_path}")

        # Nếu muốn show luôn:
        # plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        # plt.title(f"idx {idx}")
        # plt.axis("off")
        # plt.show()


if __name__ == "__main__":
    run_visualization()
