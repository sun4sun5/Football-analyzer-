import os
import tempfile
import requests as http_requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max video
mp_pose = mp.solutions.pose

CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')

@app.route('/')
def home():
    return render_template('index.html')

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
    def try_detect(image_rgb):
        """Try pose detection with multiple scales and confidence levels."""
        h, w = image_rgb.shape[:2]
        scales = [1.0, 1.5, 2.0, 0.75]
        confidences = [0.3, 0.2]
        for conf in confidences:
            for scale in scales:
                if scale != 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_scaled = cv2.resize(image_rgb, (new_w, new_h))
                else:
                    img_scaled = image_rgb
                with mp_pose.Pose(static_image_mode=True, min_detection_confidence=conf) as pose:
                    r = pose.process(img_scaled)
                    if r.pose_landmarks:
                        return r
        return None

    results = try_detect(img_rgb)
    if results is None:
        return jsonify({'error': 'لم يُرصد جسم في الصورة — حاول رفع صورة أقرب للاعب'})
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

@app.route('/claude', methods=['POST'])
def claude_analyze():
    data = request.json
    angles = data.get('angles', {})
    skill = data.get('skill', 'تسديدة')

    skill_context = {
        'تسديدة': 'ركلة تسديدة على المرمى (shooting). الزوايا المثالية: ركبة الركل 120-150°، ورك الركل 30-60°، جذع مائل للأمام قليلاً',
        'تمرير طويل': 'تمرير طويل (long pass). الزوايا المثالية: ركبة الدعم 140-160°، ورك مستقيم، جذع مائل للخلف عند التمرير',
        'استلام': 'استلام الكرة (ball reception/control). الزوايا المثالية: ركب منخفضة 100-130° للتوازن، ورك منخفض، جذع مستقيم ومرن'
    }.get(skill, '')

    prompt = f"""أنت محلل تقني متخصص في كرة القدم. قم بتحليل زوايا جسم اللاعب التالية لمهارة: {skill}

السياق التقني: {skill_context}

زوايا المفاصل المرصودة:
• ركبة يمين: {angles.get('rightKnee', 'غير متوفر')}°
• ركبة يسار: {angles.get('leftKnee', 'غير متوفر')}°
• ورك يمين: {angles.get('rightHip', 'غير متوفر')}°
• ورك يسار: {angles.get('leftHip', 'غير متوفر')}°
• الجذع: {angles.get('trunk', 'غير متوفر')}°

قدم تحليلاً شاملاً باللغة العربية يشمل:

**1. التقييم العام**
تقييم عام للوضعية مع درجة من 10

**2. نقاط القوة ✅**
ما يقوم به اللاعب بشكل صحيح

**3. نقاط التحسين ⚠️**
المفاصل التي تحتاج تعديل وكيفية التعديل

**4. تمارين مقترحة 🏋️**
3 تمارين عملية لتحسين الأداء

اجعل التحليل واضحاً ومفيداً للاعب أو المدرب."""

    if not CLAUDE_API_KEY:
        return jsonify({'error': 'مفتاح API غير مضبوط — أضف CLAUDE_API_KEY في إعدادات Railway'}), 500

    try:
        response = http_requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': CLAUDE_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-opus-4-5',
                'max_tokens': 1500,
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=30
        )
        result = response.json()
        if 'content' in result:
            return jsonify({'analysis': result['content'][0]['text']})
        else:
            error_msg = result.get('error', {}).get('message', str(result))
            return jsonify({'error': f'خطأ من Claude API: {error_msg}'}), 500
    except Exception as e:
        return jsonify({'error': f'خطأ في الاتصال بـ Claude: {str(e)}'}), 500

@app.route('/detect-kick', methods=['POST'])
def detect_kick():
    if 'video' not in request.files:
        return jsonify({'error': 'لم يتم إرسال فيديو'}), 400

    video_file = request.files['video']
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        min_dist = float('inf')
        best_time = 0.0
        ball_found = False

        # Sample every 3rd frame for speed
        step = max(1, round(fps / 10))  # ~10 checks per second

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4) as pose:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % step != 0:
                    frame_num += 1
                    continue

                time_sec = frame_num / fps
                h, w = frame.shape[:2]

                # Ball detection via HoughCircles
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
                circles = cv2.HoughCircles(
                    gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                    param1=60, param2=28, minRadius=8, maxRadius=min(w, h) // 6
                )

                if circles is not None:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        # Use ankles + foot index landmarks as foot points
                        foot_indices = [27, 28, 29, 30, 31, 32]
                        foot_points = [(lm[i].x * w, lm[i].y * h)
                                       for i in foot_indices if lm[i].visibility > 0.3]

                        if foot_points:
                            circles_arr = np.round(circles[0]).astype(int)
                            for (bx, by, br) in circles_arr:
                                for (fx, fy) in foot_points:
                                    dist = np.hypot(bx - fx, by - fy) - br
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_time = time_sec
                                        ball_found = True

                frame_num += 1

        cap.release()

        if not ball_found:
            return jsonify({'error': 'لم يتم اكتشاف الكرة في الفيديو'})

        return jsonify({'kick_time': round(best_time, 1)})

    except Exception as e:
        return jsonify({'error': f'خطأ في المعالجة: {str(e)}'}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
