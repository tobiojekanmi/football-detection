import os
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Literal, Optional

from modules.loss import CIoULoss


@dataclass
class TrainerConfig:
    """
    Configuration for model optimization and checkpointing.
    """

    # Device type for training: "cpu", "mps", or "cuda"
    device_type: str = "cpu"

    # Optimizer hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Number of training epochs
    num_epochs: int = 50

    # Evaluate model on both training and validation data every N epochs
    evaluate_every: int = 1

    # Save model checkpoint every N epochs
    save_every: int = 5

    # Whether to use a learning rate scheduler
    use_scheduler: bool = True

    # StepLR: reduce LR every `scheduler_step_size` epochs
    scheduler_step_size: int = 10

    # StepLR decay factor
    scheduler_gamma: float = 0.5

    # Loss function to use for bounding box regression
    # Options: "mse", "smooth_l1", "ciou"
    loss_type: Literal["mse", "smooth_l1", "ciou"] = "mse"


class Trainer:
    """
    Trainer class for training and validation.
    """

    def __init__(
        self,
        train_loader,
        valid_loader,
        model: torch.nn.Module,
        config: TrainerConfig,
        writer: SummaryWriter | None = None,
        model_path: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.writer = writer
        self.model_path = model_path

        self.device = torch.device(config.device_type)

        # ---- Loss, Optimizer, Scheduler ----
        if config.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif config.loss_type == "smooth_l1":
            self.criterion = torch.nn.SmoothL1Loss()
        elif config.loss_type == "ciou":
            self.criterion = CIoULoss()
        else:
            raise ValueError(
                f"Unknown loss type: {config.loss_type}. Loss should  \
                    be one of 'mse', 'smooth_l1', or 'ciou'."
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        if config.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.scheduler_step_size,
                gamma=config.scheduler_gamma,
            )

        self.model.to(self.device)

    def train_step(self):
        """
        Single training step.
        """
        self.model.train()
        running_loss = 0

        for _, images, bboxes in self.train_loader:
            images = images.to(self.device)
            bboxes = bboxes.to(self.device)

            preds = self.model(images)
            loss = self.criterion(preds, bboxes)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def valid_step(self):
        """
        Single validation step.
        """
        self.model.eval()
        running_loss = 0

        with torch.inference_mode():
            for _, images, bboxes in self.valid_loader:
                images = images.to(self.device)
                bboxes = bboxes.to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, bboxes)
                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.valid_loader.dataset)
        return epoch_loss

    def save_model(self, model_path: str, epoch: int):
        """
        Save model checkpoint.
        """
        filename = f"epoch_{epoch+1}.pth"
        save_path = os.path.join(model_path, filename)
        torch.save(self.model.state_dict(), save_path)

    def train(self):
        """
        Main training loop.
        """
        print("Starting training...\n")

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_step()

            # Evaluate according to schedule
            if (epoch == 0) or ((epoch + 1) % self.config.evaluate_every == 0):
                valid_loss = self.valid_step()
            else:
                valid_loss = None

            # Log to tensorboard if writer is provided
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                if valid_loss is not None:
                    self.writer.add_scalar("Loss/valid", valid_loss, epoch)
                self.writer.add_scalar(
                    "Learning_Rate", self.optimizer.param_groups[0]["lr"], epoch
                )

            # Print summary
            if valid_loss is not None:
                print(
                    f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                    f"- Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} "
                    f"- LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                    f"- Train Loss: {train_loss:.4f} "
                    f"- LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # LR Scheduler
            if self.config.use_scheduler:
                self.scheduler.step()

            # Save model
            if self.model_path is not None and self.config.save_every is not None:
                current_epoch = epoch + 1
                if (
                    current_epoch % self.config.save_every == 0
                    or current_epoch == self.config.num_epochs
                ):
                    self.save_model(self.model_path, epoch)

        print("\nTraining complete.")
