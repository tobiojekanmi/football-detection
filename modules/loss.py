import torch
import torch.nn as nn
import math


class CIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        pred:   (N, 4) - [x1, y1, x2, y2]
        target: (N, 4) - [x1, y1, x2, y2]
        """

        px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Intersection
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Areas
        area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        area_t = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)

        union = area_p + area_t - inter_area + self.eps
        iou = inter_area / union

        # Enclosing box
        c_x1 = torch.min(px1, tx1)
        c_y1 = torch.min(py1, ty1)
        c_x2 = torch.max(px2, tx2)
        c_y2 = torch.max(py2, ty2)
        c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + self.eps

        # Center distance
        p_cx = (px1 + px2) / 2
        p_cy = (py1 + py2) / 2
        t_cx = (tx1 + tx2) / 2
        t_cy = (ty1 + ty2) / 2
        center_dist = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

        # Aspect ratio term
        w_pred = (px2 - px1).clamp(self.eps)
        h_pred = (py2 - py1).clamp(self.eps)
        w_tgt = (tx2 - tx1).clamp(self.eps)
        h_tgt = (ty2 - ty1).clamp(self.eps)

        v = (4 / math.pi**2) * torch.pow(
            torch.atan(w_tgt / h_tgt) - torch.atan(w_pred / h_pred), 2
        )

        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        ciou = iou - (center_dist / c_diag + alpha * v)
        loss = 1 - ciou  # convert similarity to loss

        return loss.mean()
