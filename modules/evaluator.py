import torch
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluatorConfig:
    """
    Configuration for model evaluation on bounding box predictions.
    """

    # IoU threshold used to determine correct detections
    iou_threshold: float = 0.5

    # Device type for evaluation: "cpu", "mps", or "cuda"
    device_type: str = "cpu"


class Evaluator:
    """
    Evaluator class for computing mean IoU and mean accuracy metrics.
    """

    def __init__(self, model: torch.nn.Module, config: EvaluatorConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device_type)
        self.model.to(self.device)
        self.model.eval()

    def calculate_iou(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate Intersection over Union (IoU) between predicted and target boxes.

        Args:
            pred_boxes: Tensor of shape (N, 4) in format (xmin, ymin, xmax, ymax)
            target_boxes: Tensor of shape (N, 4) in format (xmin, ymin, xmax, ymax)

        Returns:
            IoU scores of shape (N,)
        """
        # Calculate intersection areas
        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Calculate union areas
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (
            pred_boxes[:, 3] - pred_boxes[:, 1]
        )
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (
            target_boxes[:, 3] - target_boxes[:, 1]
        )
        union = pred_area + target_area - intersection

        # Compute IoU (1e-6 helps avaoid division by zero)
        iou = intersection / (union + 1e-6)
        return iou

    def calculate_accuracy(
        self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor
    ) -> float:
        """
        Calculate accuracy based on IoU threshold.

        Args:
            pred_boxes: Tensor of shape (N, 4)
            target_boxes: Tensor of shape (N, 4)

        Returns:
            Accuracy score
        """
        iou_scores = self.calculate_iou(pred_boxes, target_boxes)
        accuracy = (iou_scores > self.config.iou_threshold).float().mean().item()
        return accuracy

    def evaluate(self, dataloader) -> Dict[str, float]:
        """
        Evaluate model on given dataset.

        Args:
            dataloader: DataLoader to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_iou = 0.0
        total_accuracy = 0.0
        total_samples = 0

        with torch.no_grad():
            for _, images, target_boxes in dataloader:
                images = images.to(self.device)
                target_boxes = target_boxes.to(self.device)

                pred_boxes = self.model(images)

                # Calculate metrics for this batch
                batch_iou = self.calculate_iou(pred_boxes, target_boxes).mean().item()
                batch_accuracy = self.calculate_accuracy(pred_boxes, target_boxes)
                batch_size = images.size(0)

                total_iou += batch_iou * batch_size
                total_accuracy += batch_accuracy * batch_size
                total_samples += batch_size

        mean_iou = total_iou / total_samples
        mean_accuracy = total_accuracy / total_samples

        return {"mean_iou": mean_iou, "mean_accuracy": mean_accuracy}
