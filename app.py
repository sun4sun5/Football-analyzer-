import os
import re
import time
import tempfile
import requests as http_requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import mediapipe as mp
import numpy as np
import cv2
import base64
from google import genai
from google.genai import types as genai_types

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max video
mp_pose = mp.solutions.pose

CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
gemini_client  = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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

def resize_for_claude(image_b64, max_width=800):
    """Resize image to keep Claude API tokens reasonable."""
    try:
        img_data = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8')
    except Exception:
        return image_b64

@app.route('/claude', methods=['POST'])
def claude_analyze():
    data = request.json
    angles   = data.get('angles', {})
    skill    = data.get('skill', 'تسديدة')
    image_b64 = data.get('image', None)   # صورة اللاعب (اختياري)

    skill_context = {
        'تسديدة':     'ركلة تسديدة (instep kick) — مصدر: Kellis & Katis 2007, Lees & Nolan 1998. الزوايا المثالية عند الملامسة: ركبة الركل 130-165° (امتداد قوي)، ركبة الدعم 138-154° (انحناء خفيف 26-42°)، ورك الركل 150-160° (انحناء أمامي 20-30°)، الجذع مائل للخلف 13-17° عن العمودي لتوليد القوة',
        'تمرير طويل': 'تمرير طويل (lofted instep pass) — مصدر: Lees et al. 2010, Kellis & Katis 2007. الزوايا المثالية: ركبة الركل 130-160° (مشابه للتسديدة بسرعة أقل)، ركبة الدعم 138-154°، ورك الركل 150-160°، الجذع مائل للخلف 10-15° (أقل من التسديدة) لرفع الكرة',
        'استلام':     'استلام الكرة (ball reception/trapping) — مصدر: Nagasaki et al. ISBS, Lees & Nolan 1998. الزوايا المثالية: ركبة الاستلام 130-160° (انحناء 20-50° لامتصاص الكرة)، ركبة الدعم 150-165°، ورك الاستلام 140-160°، الجذع مائل للأمام 5-15° نحو الكرة للتحكم'
    }.get(skill, '')

    prompt = f"""أنت محلل تقني متخصص في كرة القدم. قم بتحليل وضعية اللاعب في الصورة لمهارة: {skill}

السياق التقني: {skill_context}

زوايا المفاصل المحسوبة تلقائياً:
• ركبة يمين: {angles.get('rightKnee', 'غير متوفر')}°
• ركبة يسار: {angles.get('leftKnee', 'غير متوفر')}°
• ورك يمين: {angles.get('rightHip', 'غير متوفر')}°
• ورك يسار: {angles.get('leftHip', 'غير متوفر')}°
• الجذع: {angles.get('trunk', 'غير متوفر')}°

استخدم الصورة والزوايا معاً لتقديم تحليل شامل باللغة العربية يشمل:

**1. التقييم العام**
تقييم عام للوضعية مع درجة من 10

**2. نقاط القوة ✅**
ما يقوم به اللاعب بشكل صحيح (استند للصورة والزوايا)

**3. نقاط التحسين ⚠️**
المفاصل التي تحتاج تعديل وكيفية التعديل

**4. تمارين مقترحة 🏋️**
3 تمارين عملية لتحسين الأداء

اجعل التحليل واضحاً ومفيداً للاعب أو المدرب."""

    if not CLAUDE_API_KEY:
        return jsonify({'error': 'مفتاح API غير مضبوط — أضف CLAUDE_API_KEY في إعدادات Railway'}), 500

    # بناء المحتوى — صورة + نص إذا توفرت الصورة
    if image_b64:
        resized = resize_for_claude(image_b64)
        message_content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": resized}},
            {"type": "text",  "text": prompt}
        ]
    else:
        message_content = prompt

    try:
        response = http_requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': CLAUDE_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 1000,
                'messages': [{'role': 'user', 'content': message_content}]
            },
            timeout=25
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

        best_time = 0.0
        ball_found = False
        step = max(1, round(fps / 20))
        foot_positions = {}

        with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                          min_detection_confidence=0.15, min_tracking_confidence=0.1) as pose:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % step == 0:
                    h, w = frame.shape[:2]
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)
                    if results.pose_landmarks:
                        lm = results.pose_landmarks.landmark
                        ankles = []
                        for idx in [27, 28]:
                            if lm[idx].visibility > 0.2:
                                ankles.append((lm[idx].x * w, lm[idx].y * h, lm[idx].visibility))
                        if ankles:
                            ax, ay, _ = max(ankles, key=lambda a: a[2])
                            foot_positions[frame_num] = (ax, ay)
                frame_num += 1

        if len(foot_positions) >= 3:
            sorted_frames = sorted(foot_positions.keys())
            speeds = {}
            for i in range(1, len(sorted_frames) - 1):
                f_prev, f_curr, f_next = sorted_frames[i-1], sorted_frames[i], sorted_frames[i+1]
                px, py = foot_positions[f_prev]
                cx, cy = foot_positions[f_curr]
                nx, ny = foot_positions[f_next]
                speeds[f_curr] = (np.hypot(cx - px, cy - py) + np.hypot(nx - cx, ny - cy)) / 2

            if speeds:
                max_speed = max(speeds.values())
                threshold = max_speed * 0.6
                speed_frames = sorted(speeds.keys())
                for i in range(1, len(speed_frames) - 1):
                    f_prev, f_curr, f_next = speed_frames[i-1], speed_frames[i], speed_frames[i+1]
                    if speeds[f_curr] > speeds[f_prev] and speeds[f_curr] > speeds[f_next] and speeds[f_curr] >= threshold:
                        best_time = f_curr / fps
                        ball_found = True
                        break
                if not ball_found:
                    best_time = max(speeds, key=speeds.get) / fps
                    ball_found = True

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

        # ── Step 1: Default positions (fallback if Gemini unavailable) ──
        contact_frame_num = total_frames // 2
        prep_frame_num    = max(0, contact_frame_num - int(fps * 0.4))
        follow_frame_num  = min(total_frames - 1, contact_frame_num + int(fps * 0.4))

        cap.release()

        # ── Step 2: Use Gemini to identify the 3 phases from the full video ──

        if gemini_client:
            try:
                duration = total_frames / fps
                # Upload video to Gemini Files API
                with open(tmp_path, 'rb') as f:
                    video_file = gemini_client.files.upload(
                        file=f,
                        config=genai_types.UploadFileConfig(mime_type='video/mp4')
                    )
                # Wait for processing (max 40s)
                for _ in range(20):
                    video_file = gemini_client.files.get(name=video_file.name)
                    if video_file.state.name == 'ACTIVE':
                        break
                    time.sleep(2)

                if video_file.state.name == 'ACTIVE':
                    prompt = (
                        f"هذا فيديو لتسديدة كرة قدم مدته {duration:.1f} ثانية. "
                        "حدد بدقة الثانية التي تمثل كل مرحلة:\n"
                        "1. التحضير: لحظة ذروة الـ backswing (القدم في أعلى نقطة خلف الجسم)\n"
                        "2. الملامسة: اللحظة الفعلية التي تلمس فيها القدم الكرة\n"
                        "3. المتابعة: ذروة الـ follow-through (القدم في أعلى نقطة أمام الجسم)\n"
                        "أجب بهذا الشكل الحرفي فقط (أرقام عشرية):\n"
                        "PREP=<ثانية>\nCONTACT=<ثانية>\nFOLLOW=<ثانية>"
                    )
                    response = gemini_client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=[
                            genai_types.Part.from_uri(file_uri=video_file.uri, mime_type='video/mp4'),
                            prompt
                        ]
                    )
                    text = response.text
                    pm = re.search(r'PREP=([\d.]+)', text)
                    cm = re.search(r'CONTACT=([\d.]+)', text)
                    fm = re.search(r'FOLLOW=([\d.]+)', text)
                    if pm:
                        prep_frame_num = min(total_frames-1, max(0, int(float(pm.group(1)) * fps)))
                    if cm:
                        contact_frame_num = min(total_frames-1, max(0, int(float(cm.group(1)) * fps)))
                    if fm:
                        follow_frame_num = min(total_frames-1, max(0, int(float(fm.group(1)) * fps)))

                try:
                    gemini_client.files.delete(name=video_file.name)
                except Exception:
                    pass

            except Exception:
                pass  # fallback to default values

        ball_found = True

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
        'تسديدة':     'ركلة تسديدة (instep kick) — مصدر: Kellis & Katis 2007, Lees & Nolan 1998. الزوايا المثالية عند الملامسة: ركبة الركل 130-165° (امتداد قوي)، ركبة الدعم 138-154° (انحناء خفيف 26-42°)، ورك الركل 150-160° (انحناء أمامي 20-30°)، الجذع مائل للخلف 13-17° عن العمودي لتوليد القوة',
        'تمرير طويل': 'تمرير طويل (lofted instep pass) — مصدر: Lees et al. 2010, Kellis & Katis 2007. الزوايا المثالية: ركبة الركل 130-160° (مشابه للتسديدة بسرعة أقل)، ركبة الدعم 138-154°، ورك الركل 150-160°، الجذع مائل للخلف 10-15° (أقل من التسديدة) لرفع الكرة',
        'استلام':     'استلام الكرة (ball reception/trapping) — مصدر: Nagasaki et al. ISBS, Lees & Nolan 1998. الزوايا المثالية: ركبة الاستلام 130-160° (انحناء 20-50° لامتصاص الكرة)، ركبة الدعم 150-165°، ورك الاستلام 140-160°، الجذع مائل للأمام 5-15° نحو الكرة للتحكم'
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
انظر إلى الصور الثلاث المرفقة (التحضير، الملامسة، المتابعة) واستخدمها مع الزوايا المحسوبة للتحليل.

السياق التقني: {skill_context}

زوايا الجسم في كل مرحلة:

🏃 التحضير (الثانية {prep.get('time', '—')}): {angles_text(prep)}
⚽ الملامسة (الثانية {contact.get('time', '—')}): {angles_text(contact)}
🎯 المتابعة (الثانية {followthrough.get('time', '—')}): {angles_text(followthrough)}

قدم تحليلاً شاملاً للحركة الكاملة باللغة العربية يشمل:

**1. التقييم العام للحركة الكاملة**
تقييم عام مع درجة من 10

**2. تحليل كل مرحلة** (استند للصور والزوايا)
• التحضير: ما هو صحيح وما يحتاج تعديل
• الملامسة: جودة وضعية اللحظة الحاسمة
• المتابعة: اكتمال الحركة والتوازن

**3. نقاط القوة ✅**
أبرز ما يقوم به اللاعب بشكل صحيح

**4. نقاط التحسين ⚠️**
المراحل والمفاصل التي تحتاج تعديلاً مع وصف دقيق

**5. تمارين مقترحة 🏋️**
3 تمارين عملية لتحسين تسلسل الحركة

اجعل التحليل واضحاً ومفيداً للاعب أو المدرب."""

    if not CLAUDE_API_KEY:
        return jsonify({'error': 'مفتاح API غير مضبوط — أضف CLAUDE_API_KEY في إعدادات Railway'}), 500

    # بناء المحتوى — صورة الملامسة فقط + النص (لتجنب timeout)
    message_content = []
    contact_thumb = phases.get('contact', {}).get('thumbnail') or \
                    phases.get('preparation', {}).get('thumbnail') or \
                    phases.get('followthrough', {}).get('thumbnail')
    if contact_thumb:
        message_content.append({"type": "text", "text": "صورة لحظة الملامسة:"})
        message_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": contact_thumb}
        })
    message_content.append({"type": "text", "text": prompt})

    try:
        response = http_requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': CLAUDE_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 1200,
                'messages': [{'role': 'user', 'content': message_content}]
            },
            timeout=25
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
