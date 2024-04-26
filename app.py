import csv
from flask import Flask, render_template, Response
from hand_gesture import gen_video_feed

app = Flask(__name__)

# Read the labels 
labels = []

with open('labels.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        labels.append(row)

@app.route('/')
def index():
    return render_template('index.html', labels=labels)

@app.route('/video_feed')
def video_feed():
    return Response(gen_video_feed(), content_type='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
