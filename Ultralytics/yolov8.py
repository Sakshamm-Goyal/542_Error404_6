import os
import datetime
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = YOLO('yolov8m.pt')


batch_size = 8
lower_resolution = (540, 380)

start_time = datetime.datetime.now()


results = [datetime.datetime.now(), model.track(source=r"hall2.mp4", show=True, tracker="bytetrack.yaml", batch=batch_size, imgsz=lower_resolution)]

end_time = datetime.datetime.now()
processing_time = end_time - start_time
print(f"Processing Time: {processing_time}")


print(results)
