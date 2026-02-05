for m in yolo26n yolo26m yolo26x rtdetr-l rtdetr-x yolov5nu yolov5mu yolov5xu; do
    for s in 384 512 640 768; do
        yolo detect val model=weights/${m}.pt imgsz=$s data=configs/coco.yaml batch=1
    done
done