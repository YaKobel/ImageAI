from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(exec_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

list = detector.detectCustomObjectsFromImage(
    input_image=os.path.join(exec_path, "objects.jpg"),
    output_image_path=os.path.join(exec_path, "new_objects.jpg"),
    minimum_percentage_probability=70,
    display_percentage_probability=True,
    display_object_name=False
)


from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "traffic.mp4"),
    output_file_path=os.path.join(execution_path, "traffic_detected"),
    frames_per_second=20,
    log_progress=True
)

print(video_path)