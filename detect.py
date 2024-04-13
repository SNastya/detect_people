import cv2
import torch

def main():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt') 
    model.conf = 0.21
    
    cap = cv2.VideoCapture("in/crowd.mp4") 

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('out/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        results = model(frame, size=640)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            if labels[i] == 0:
                row = cord[i]
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                # label = results.names[int(labels[i])] + str(row[4])[6:]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    
        cv2.imshow('YOLOv5 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
