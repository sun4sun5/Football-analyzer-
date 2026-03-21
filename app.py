from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    cos = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return round(np.degrees(np.arccos(np.clip(cos, -1, 1))))

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(img_rgb)
        if not results.pose_landmarks:
            return jsonify({'error': 'لم يُرصد جسم في الصورة'})
        lm = results.pose_landmarks.landmark
        def p(i): return [lm[i].x, lm[i].y]
        angles = {
            'rightKnee': calculate_angle(p(24), p(26), p(28)),
            'leftKnee': calculate_angle(p(23), p(25), p(27)),
            'rightHip': calculate_angle(p(12), p(24), p(26)),
            'leftHip': calculate_angle(p(11), p(23), p(25)),
            'trunk': calculate_angle(p(11), p(23), p(24))
        }
        return jsonify({'angles': angles})

if __name__ == '__main__':
    app.run()
