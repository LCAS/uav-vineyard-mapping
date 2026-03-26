"""
Patch for supervision library compatibility with RF-DETR training.

RF-DETR codebase uses supervision.xyxy_to_xywh which doesn't exist in newer versions.
This module adds the missing function to supervision.
"""

import supervision as sv
import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from xyxy format to xywh format.
    
    Args:
        boxes: Array of shape (N, 4) with boxes in xyxy format  [x1, y1, x2, y2]
        
    Returns:
        Array of shape (N, 4) with boxes in xywh format [x, y, width, height]
    """
    if boxes.size == 0:
        return boxes
    
    boxes = boxes.copy()
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = x2 - x1
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = y2 - y1
    
    return boxes


# Monkey-patch the function into supervision
if not hasattr(sv, 'xyxy_to_xywh'):
    sv.xyxy_to_xywh = xyxy_to_xywh
    print("✓ Patched supervision.xyxy_to_xywh")
