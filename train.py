from .models.ResLKA_Unet import ResLKA_Unet
from .models.ResNet50_Unet import Resnet50_Unet
from dataset import LiTS
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
from .loss import *
from .metric import compute_scores
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import os
import shutil
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="train unet")
    parser.add_argument("--epochs", "-e", type=int, default=5)
    parser.add_argument("--batch_size", "-b", type=int, default=2)
    parser.add_argument("--root", "-r", type=str, default=r"D:\DLFS\Unet\sample")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--lowerbound", "-lb", type=int, default=0)
    parser.add_argument("--upperbound", "-ub", type=int, default=100)
    parser.add_argument("--bce_weight", "-bw", type=float, default=0.5)
    parser.add_argument("--dice_weight", "-dw", type=float, default=0.5)
    parser.add_argument("--focal_weight", "-fw", type=float, default=0.5)
    parser.add_argument("--stage", "-s", type=int, default=1)
    return parser.parse_args()


# ====================== Trainer Class ======================
class Trainer:
    def __init__(self, args, model, train_loader, dev_loader, device):
        self.args = args
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        self.optimizer = Adam(model.parameters(), lr=args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

        # Logging setup
        if os.path.isdir(args.logging):
            shutil.rmtree(args.logging)
        if not os.path.isdir(args.trained_models):
            os.mkdir(args.trained_models)
        self.writer = SummaryWriter(args.logging)

        # checkpoint
        self.best_iou = 0.0
        self.start_epoch = 0
        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)

    # -------------------- Train 1 Batch --------------------
    def _run_batch(self, images, masks):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss_value = (
            nn.BCELoss()(outputs, masks)*self.args.bce_weight
            + dice_loss(outputs, masks)*self.args.dice_weight
            + focal_loss(outputs, masks)*self.args.focal_weight
        )
        loss_value.backward()
        self.optimizer.step()
        return loss_value

    # -------------------- Train 1 Epoch --------------------
    def _train_epoch(self, epoch):
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{self.args.epochs}. Learning rate: {current_lr:.9f}")
        self.model.train()
        process_bar = tqdm(self.train_loader, colour="cyan")
        num_iters = len(self.train_loader)
        for iter, (images, masks) in enumerate(process_bar):
            images, masks = images.to(self.device), masks.to(self.device)
            loss_value = self._run_batch(images, masks)

            process_bar.set_description(
                f"[Train] Epoch {epoch + 1}/{self.args.epochs}. Iter {iter + 1}/{num_iters}. Loss {loss_value:.3f}"
            )
            self.writer.add_scalar("Train/Loss", loss_value.item(), epoch * num_iters + iter)

    # -------------------- Validation --------------------
    def _validate(self, epoch):
        self.model.eval()
        all_predictions, all_masks = [], []
        dev_process_bar = tqdm(self.dev_loader, colour="cyan")
        num_iters = len(self.dev_loader)

        with torch.no_grad():
            for dev_iter, (images, masks) in enumerate(dev_process_bar):
                images, masks = images.to(self.device), masks.to(self.device)
                pred = self.model(images)

                dev_loss_value = (
                    nn.BCELoss()(pred, masks)*self.args.bce_weight
                    + dice_loss(pred, masks)*self.args.dice_weight
                    + focal_loss(pred, masks)*self.args.focal_weight
                )

                dev_process_bar.set_description(
                    f"[Val] Epoch {epoch + 1}/{self.args.epochs}. Iter {dev_iter + 1}/{num_iters}. Loss {dev_loss_value:.3f}"
                )
                self.writer.add_scalar("Dev/Loss", dev_loss_value.item(), epoch * num_iters + dev_iter)

                all_predictions.extend((pred > 0.55).long().cpu().numpy())
                all_masks.extend(masks.cpu().numpy())

        score = compute_scores(all_predictions, all_masks)
        iou_score = score["iou"]
        print(f"IOU: {iou_score:.4f}")
        self.writer.add_scalar("Val/IOU", iou_score, epoch)
        self.scheduler.step()

        return iou_score

    # -------------------- Save Checkpoint --------------------
    def _save_checkpoint(self, epoch, best=False, iou_score=None):
        checkpoint = {
            "epoch": epoch + 1,
            "best_iou": self.best_iou if not best else iou_score,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        name = "best_model.pt" if best else "last_model.pt"
        torch.save(checkpoint, f"{self.args.trained_models}/{name}")

    # -------------------- Load Checkpoint --------------------
    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.start_epoch = checkpoint["epoch"]
        self.best_iou = checkpoint["best_iou"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Loaded checkpoint from {path} (epoch {self.start_epoch}, best_iou={self.best_iou:.3f})")

    # -------------------- Main Loop --------------------
    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self._train_epoch(epoch)
            iou_score = self._validate(epoch)

            # save last checkpoint
            self._save_checkpoint(epoch)

            # save best checkpoint
            if iou_score > self.best_iou:
                self.best_iou = iou_score
                self._save_checkpoint(epoch, best=True, iou_score=iou_score)
                print(f"New best IOU: {iou_score:.4f}")


# ====================== Data Loader Function ======================
def prepare_dataloader(args):
    transform = Compose([ToTensor()])
    target_transform = Compose([ToTensor()])

    train_dataset = LiTS(
        root=args.root,
        train=True,
        lowerbound=args.lowerbound,
        upperbound=args.upperbound,
        transform=transform,
        target_transform=target_transform,
        stage=args.stage,
    )
    dev_dataset = LiTS(
        root=args.root,
        dev=True,
        lowerbound=args.lowerbound,
        upperbound=args.upperbound,
        transform=transform,
        target_transform=target_transform,
        stage=args.stage,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, dev_loader


# ====================== Main Entry ======================
def main():
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  
    else:
        device = torch.device("cpu")

    model = ResLKA_Unet().to(device)
    train_loader, dev_loader = prepare_dataloader(args)

    trainer = Trainer(args, model, train_loader, dev_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()