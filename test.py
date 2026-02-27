import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import os
import json
from .dataset import LiTS
from .models.ResLKA_Unet import ResLKA_Unet
from .models.ResNet50_Unet import Resnet50_Unet
from .loss import *
from .metric import compute_scores


def get_args():
    parser = ArgumentParser(description='test unet')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--root', '-r', type=str, default=r'D:\DLFS\Unet\sample')
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument('--lowerbound', '-lb', type=int, default=0)
    parser.add_argument('--upperbound', '-ub', type=int, default=100)
    parser.add_argument('--json_dir', '-jd', type=str, default="results")
    parser.add_argument("--stage", "-s", type=int, default=1)
    args = parser.parse_args()
    return args


class Tester:
    def __init__(self, model, dataloader, device, json_dir):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.json_dir = json_dir

        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir, exist_ok=True)

    def _run_batch(self, images, masks):
        images = images.to(self.device)
        masks = masks.to(self.device)

        with torch.no_grad():
            preds = self.model(images)

        preds = (preds > 0.5).long().cpu().numpy()
        masks = masks.cpu().numpy()
        return preds, masks

    def test(self):
        self.model.eval()
        all_predictions, all_masks = [], []

        progress_bar = tqdm(self.dataloader, desc="Testing", unit="batch")
        for images, masks in progress_bar:
            preds, gt = self._run_batch(images, masks)
            all_predictions.extend(preds)
            all_masks.extend(gt)

        scores = compute_scores(all_predictions, all_masks)
        print("\n=== Test Metrics ===")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        output_file = os.path.join(self.json_dir, "scores.json")
        with open(output_file, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"Scores saved to: {output_file}")


def main():
    args = get_args()

    transform = Compose([ToTensor()])
    target_transform = Compose([ToTensor()])

    test_dataset = LiTS(
        root=args.root,
        train=False,
        lowerbound=args.lowerbound,
        upperbound=args.upperbound,
        transform=transform,
        target_transform=target_transform,
        stage=args.stage,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = ResLKA_Unet()
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tester = Tester(model, test_loader, device, args.json_dir)
    tester.test()


if __name__ == "__main__":
    main()
