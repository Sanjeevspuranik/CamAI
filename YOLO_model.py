# Example using Ultralytics YOLOv8 API (adapt for YOLOv26)
from ultralytics import YOLO


class YOLOInference:
    """
    Wrapper for YOLO object detection.
    """

    def __init__(self, model_path="yolo26n.pt"):
        self.model = YOLO(model_path)

    def infer(self, frame) -> list[str]:
        """
        Run YOLO detection on a frame (numpy array).
        Returns list of detected class names.
        """
        results = self.model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(r.names[cls_id])
        return detections
