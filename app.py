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

def enhance_image(image_rgb):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

def try_detect(image_rgb):
    h, w = image_rgb.shape[:2]
    scales = [2.0, 3.0, 1.5, 4.0, 1.0]
    confidences = [0.3, 0.15, 0.05]
    complexities = [2, 1]
    images_to_try = [image_rgb, enhance_image(image_rgb)]
    for img_variant in images_to_try:
        for complexity in complexities:
            for conf in confidences:
                for scale in scales:
                    if scale != 1.0:
                        new_w, new_h = int(w * scale), int(h * scale)
                        img_scaled = cv2.resize(img_variant, (new_w, new_h),
                                                interpolation=cv2.INTER_CUBIC)
                    else:
                        img_scaled = img_variant
                    with mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=complexity,
                        min_detection_confidence=conf
                    ) as pose:
                        r = pose.process(img_scaled)
                        if r.pose_landmarks:
                            return r
    return None

def extract_angles(results):
    lm = results.pose_landmarks.landmark
    def p(i): return [lm[i].x, lm[i].y]
    return {
        'rightKnee': calculate_angle(p(24), p(26), p(28)),
        'leftKnee':  calculate_angle(p(23), p(25), p(27)),
        'rightHip':  calculate_angle(p(12), p(24), p(26)),
        'leftHip':   calculate_angle(p(11), p(23), p(25)),
        'trunk':     calculate_angle(p(11), p(23), p(24))
    }

def frame_to_base64(frame_bgr, max_width=400):
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * scale)))
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = try_detect(img_rgb)
    if results is None:
        return jsonify({'error': 'لم يُرصد جسم في الصورة — حاول رفع صورة أقرب للاعب'})
    return jsonify({'angles': extract_angles(results)})

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

        step = max(1, round(fps / 10))

        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.15, min_tracking_confidence=0.1) as pose:
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

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
                circles = cv2.HoughCircles(
                    gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                    param1=60, param2=28, minRadius=8, maxRadius=min(w, h) // 6
                )

                if circles is not None:
                    frame_up = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                    img_rgb = cv2.cvtColor(frame_up, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)
                    circles = circles * 2 if circles is not None else circles

                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        foot_indices = [27, 28, 29, 30, 31, 32]
                        foot_points = [(lm[i].x * w * 2, lm[i].y * h * 2)
                                       for i in foot_indices if lm[i].visibility > 0.1]

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


@app.route('/analyze-video', methods=['POST'])
def analyze_video():
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

        # ── Step 1: Find contact frame ──
        min_dist = float('inf')
        contact_frame_num = total_frames // 2
        ball_found = False
        step = max(1, round(fps / 10))

        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.15, min_tracking_confidence=0.1) as pose:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % step == 0:
                    h, w = frame.shape[:2]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
                    circles = cv2.HoughCircles(
                        gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                        param1=60, param2=28, minRadius=8, maxRadius=min(w, h) // 6
                    )
                    if circles is not None:
                        frame_up = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                        img_rgb = cv2.cvtColor(frame_up, cv2.COLOR_BGR2RGB)
                        results = pose.process(img_rgb)
                        circles_scaled = circles * 2
                        if results.pose_landmarks:
                            lm = results.pose_landmarks.landmark
                            foot_indices = [27, 28, 29, 30, 31, 32]
                            foot_points = [(lm[i].x * w * 2, lm[i].y * h * 2)
                                           for i in foot_indices if lm[i].visibility > 0.1]
                            if foot_points:
                                circles_arr = np.round(circles_scaled[0]).astype(int)
                                for (bx, by, br) in circles_arr:
                                    for (fx, fy) in foot_points:
                                        dist = np.hypot(bx - fx, by - fy) - br
                                        if dist < min_dist:
                                            min_dist = dist
                                            contact_frame_num = frame_num
                                            ball_found = True
                frame_num += 1

        cap.release()

        # ── Step 2: Define 3 phase frame numbers ──
        prep_frame_num    = max(0, contact_frame_num - int(fps * 0.5))
        follow_frame_num  = min(total_frames - 1, contact_frame_num + int(fps * 0.3))

        # ── Step 3: Seek to each phase and analyze ──
        cap = cv2.VideoCapture(tmp_path)
        phases = {}
        phase_defs = [
            ('preparation',  prep_frame_num,    'التحضير'),
            ('contact',      contact_frame_num,  'الملامسة'),
            ('followthrough', follow_frame_num,  'المتابعة'),
        ]

        for phase_key, f_num, phase_name in phase_defs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
            ret, frame = cap.read()
            if not ret:
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = try_detect(img_rgb)
            if result:
                phases[phase_key] = {
                    'angles':    extract_angles(result),
                    'thumbnail': frame_to_base64(frame),
                    'time':      round(f_num / fps, 1),
                    'name':      phase_name
                }

        cap.release()

        if not phases:
            return jsonify({'error': 'لم يتم اكتشاف وضعية اللاعب في أي مرحلة من الفيديو'})

        return jsonify({
            'phases':       phases,
            'ball_found':   ball_found,
            'contact_time': round(contact_frame_num / fps, 1)
        })

    except Exception as e:
        return jsonify({'error': f'خطأ في معالجة الفيديو: {str(e)}'}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/claude-video', methods=['POST'])
def claude_video():
    data = request.json
    phases = data.get('phases', {})
    skill  = data.get('skill', 'تسديدة')

    skill_context = {
        'تسديدة':     'ركلة تسديدة على المرمى (shooting). الزوايا المثالية: ركبة الركل 120-150°، ورك الركل 30-60°، جذع مائل للأمام قليلاً',
        'تمرير طويل': 'تمرير طويل (long pass). الزوايا المثالية: ركبة الدعم 140-160°، ورك مستقيم، جذع مائل للخلف عند التمرير',
        'استلام':     'استلام الكرة (ball reception/control). الزوايا المثالية: ركب منخفضة 100-130° للتوازن، ورك منخفض، جذع مستقيم ومرن'
    }.get(skill, '')

    def angles_text(phase_data):
        if not phase_data:
            return 'غير متوفر'
        a = phase_data.get('angles', {})
        return (f"ركبة يمين: {a.get('rightKnee','—')}°، ركبة يسار: {a.get('leftKnee','—')}°، "
                f"ورك يمين: {a.get('rightHip','—')}°، ورك يسار: {a.get('leftHip','—')}°، "
                f"الجذع: {a.get('trunk','—')}°")

    prep          = phases.get('preparation', {})
    contact       = phases.get('contact', {})
    followthrough = phases.get('followthrough', {})

    prompt = f"""أنت محلل تقني متخصص في كرة القدم. قم بتحليل حركة اللاعب الكاملة لمهارة: {skill}

السياق التقني: {skill_context}

تم رصد زوايا الجسم في ثلاث مراحل زمنية متتالية:

🏃 مرحلة التحضير (الثانية {prep.get('time', '—')}):
{angles_text(prep)}

⚽ مرحلة الملامسة (الثانية {contact.get('time', '—')}):
{angles_text(contact)}

🎯 مرحلة المتابعة (الثانية {followthrough.get('time', '—')}):
{angles_text(followthrough)}

قدم تحليلاً شاملاً للحركة الكاملة باللغة العربية يشمل:

**1. التقييم العام للحركة الكاملة**
تقييم عام للحركة من التحضير إلى المتابعة مع درجة من 10

**2. تحليل كل مرحلة**
• التحضير: ما هو صحيح وما يحتاج تعديل
• الملامسة: جودة وضعية اللحظة الحاسمة
• المتابعة: اكتمال الحركة والتوازن

**3. نقاط القوة ✅**
أبرز ما يقوم به اللاعب بشكل صحيح عبر الحركة كاملة

**4. نقاط التحسين ⚠️**
المفاصل والمراحل التي تحتاج تعديلاً مع وصف دقيق

**5. تمارين مقترحة 🏋️**
3 تمارين عملية تستهدف تحسين تسلسل الحركة وتدفقها

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
                'max_tokens': 2000,
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=45
        )
        result = response.json()
        if 'content' in result:
            return jsonify({'analysis': result['content'][0]['text']})
        else:
            error_msg = result.get('error', {}).get('message', str(result))
            return jsonify({'error': f'خطأ من Claude API: {error_msg}'}), 500
    except Exception as e:
        return jsonify({'error': f'خطأ في الاتصال بـ Claude: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
