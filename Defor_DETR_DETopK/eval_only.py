# eval_only.py
# ------------------------------------------------------------------------
# Eval-only script for Deformable DETR / DETopK
# Dùng engine.evaluate, KHÔNG train lại
# ------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader, SequentialSampler

from main import get_args_parser          # dùng parser y như main.py
from models import build_model
from datasets import build_dataset
from engine import evaluate
import util.misc as utils                 # để dùng collate_fn


@torch.no_grad()
def run_eval_only(args, split="val"):
    """
    Hàm riêng để evaluate:
    - split: 'train' / 'val' / 'test' (tùy bạn đã config trong datasets/coco.py)
    """
    device = torch.device(args.device)

    # 1. Build model + criterion + postprocessors
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # 2. Load checkpoint
    assert args.resume is not None, "Cần truyền --resume tới checkpoint .pth"
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[Eval-only] Loaded checkpoint from {args.resume}")

    # 3. Build dataset + dataloader cho split cần eval
    dataset = build_dataset(split, args)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,   # giống main.py
    )

    # 4. Lấy COCO API object để CocoEvaluator dùng
    base_ds = getattr(dataset, "coco", None)

    # 5. Gọi engine.evaluate
    stats, coco_evaluator = evaluate(
        model, criterion, postprocessors,
        data_loader, base_ds,
        device, args.output_dir
    )

    # 6. In kết quả COCO mAP
    if coco_evaluator is not None:
        coco_evaluator.summarize()

    print("\n[Eval-only] Metric logger stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats, coco_evaluator


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--resume", required=True,
                        help="Đường dẫn tới checkpoint .pth để eval")
    parser.add_argument("--eval_split", default="val",
                        choices=["train", "val", "test"],
                        help="Tập cần đánh giá: train / val / test (nếu đã khai báo trong coco.py)")
    args = parser.parse_args()

    run_eval_only(args, split=args.eval_split)
