import cv2
import numpy as np
import datetime

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

BLACK = (0,0,0)
BLUE = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


classesFile = "config_files/classes.txt"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    
    
modelWeights = "config_files/yolov5n.onnx"
net = cv2.dnn.readNet(modelWeights)



# source = 0 for web camera 1
source ="sample.mp4"
video_frame = cv2.VideoCapture(source)


fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
while True:
    
    _, input_image = video_frame.read()
    #input_image = imutils.resize(frame, width=600)
    total_frames = total_frames + 1
    if input_image is None:
        print("End of stream")
        break

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    #blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    
    # Sets the input to the network.
    net.setInput(blob)
    
    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    # print(outputs[0].shape)
    
    
    
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    
    # Rows.
    rows = outputs[0].shape[1]
    
    image_height, image_width = input_image.shape[:2]
    
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    
    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
    
        # Discard bad detections and continue.
        if (confidence) >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
    
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
    
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
    
                cx, cy, w, h = row[0], row[1], row[2], row[3]
    
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
    
                box = np.array([left, top, width, height])
                boxes.append(box)
    
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        #draw_label(input_image, label, left, top)
        # Get text size.
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle. 
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
        # Display text inside the rectangle.
        cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)
    
        
        #fps
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
    
        fps_text = "FPS: {:.2f}".format(fps)
    
        cv2.putText(input_image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        print("fps:", fps_text)
        #out_.write(frame)
        cv2.imshow('Output', input_image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
# =============================================================================
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
# =============================================================================

# Release everything if job is finished
video_frame.release()
#out_.release()
cv2.destroyAllWindows()

