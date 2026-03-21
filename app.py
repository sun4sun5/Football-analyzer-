import os
import requests as http_requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64

app = Flask(__name__)
CORS(app)
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

    try:
        response = http_requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': CLAUDE_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-opus-4-6',
                'max_tokens': 1500,
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=30
        )
        result = response.json()
        if 'content' in result:
            return jsonify({'analysis': result['content'][0]['text']})
        else:
            return jsonify({'error': 'خطأ في استجابة Claude', 'details': str(result)}), 500
    except Exception as e:
        return jsonify({'error': f'خطأ في الاتصال بـ Claude: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
