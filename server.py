from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

vid = cv2.VideoCapture(0)
face_detection = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

def frame_generator():
    while True:
        success, frame = vid.read()
        if success == False:
            break
        else:
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detection.detectMultiScale(frame_grey, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            frame_encode = cv2.imencode(".jpg", frame)[1]
            frame_string = frame_encode.tobytes()
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_string + b'\r\n')

@app.route("/")
def page():
    return render_template('index.html')

@app.route('/show_feed')
def show_feed():
    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)