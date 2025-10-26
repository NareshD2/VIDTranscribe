# app.py

# ================== Install Dependencies ==================
# Install these manually via requirements.txt or pip
# pip install -r requirements.txt

# ================== Imports ==================
import os
import cv2
import easyocr
import tempfile
import numpy as np
import threading
import time
import json
from flask import Flask, request, send_file, Response, jsonify, stream_with_context
from flask_cors import CORS
from deep_translator import GoogleTranslator
from PIL import ImageFont, ImageDraw, Image

# ================== Flask App ==================
app = Flask(__name__)
CORS(app, origins=[
    "https://localhost:3001",
    "http://localhost:3001",
    "https://localhost:3000",
    "http://localhost:3000"
])

# Shared state for progress updates
progress_data = {"status": "idle", "processed": 0, "total": 0, "message": ""}

# ================== Helper Functions ==================
def convert_numerals_to_arabic(text):
    devanagari_map = {'०':'0','१':'1','२':'2','३':'3','४':'4','५':'5','६':'6','७':'7','८':'8','९':'9'}
    eastern_arabic_map = {'٠':'0','١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'}
    combined_map = {**devanagari_map, **eastern_arabic_map}
    return text.translate(str.maketrans(combined_map))

def inpaint_text_area(frame, bbox):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(bbox, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

def get_font_for_lang(lang_code, size=32):
    FONT_DIR = os.path.join(os.getcwd(), "fonts")
    FONT_MAP = {
        'hi': os.path.join(FONT_DIR, "NotoSansDevanagari-Regular.ttf"),
        'te': os.path.join(FONT_DIR, "NotoSansTelugu-Regular.ttf"),
        'ta': os.path.join(FONT_DIR, "NotoSansTamil-Regular.ttf"),
        'bn': os.path.join(FONT_DIR, "NotoSansBengali-Regular.ttf"),
        'ja': os.path.join(FONT_DIR, "NotoSansJP-Regular.otf"),
        'ko': os.path.join(FONT_DIR, "NotoSansKR-Regular.otf"),
        'zh-CN': os.path.join(FONT_DIR, "NotoSansSC-Regular.otf"),
        'ar': os.path.join(FONT_DIR, "NotoSansArabic-Regular.ttf"),
    }
    default_font = os.path.join(FONT_DIR, "NotoSans-Regular.ttf")
    path = FONT_MAP.get(lang_code, default_font)
    
    if not os.path.exists(path):
        print(f"⚠️ Font not found for {lang_code}, using default.")
        path = default_font
    
    return ImageFont.truetype(path, size)


def draw_translated_text(frame, boxes_and_translations, target_language):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for box, text in boxes_and_translations:
        if not text:
            continue
        (tl, tr, br, bl) = box
        box_height = int(np.linalg.norm(np.array(tl) - np.array(bl)))
        box_width = int(np.linalg.norm(np.array(tr) - np.array(tl)))
        font_size = box_height
        font = get_font_for_lang(target_language, font_size)
        try:
            text_bbox = font.getbbox(text)
            text_w = text_bbox[2] - text_bbox[0]
        except:
            text_w, _ = font.getsize(text)
        while text_w > box_width * 0.95 and font_size > 10:
            font_size -= 2
            font = get_font_for_lang(target_language, font_size)
            try:
                text_bbox = font.getbbox(text)
                text_w = text_bbox[2] - text_bbox[0]
            except:
                text_w, _ = font.getsize(text)
        center_x = int(tl[0] + box_width / 2)
        center_y = int(tl[1] + box_height / 2)
        draw.text(
            (center_x, center_y),
            text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
            anchor="mm",
        )
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ================== Core Processing Function ==================
def process_video(input_path, output_path, src_lang, target_lang):
    global progress_data
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_data.update({"status": "processing", "processed": 0, "total": frame_count, "message": "Starting..."})

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    reader = easyocr.Reader([src_lang], gpu=False)  # Change to gpu=True if GPU available
    translator = GoogleTranslator(source="auto", target=target_lang)
    translation_cache = {}
    last_known_results = []
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        current_ocr_results = reader.readtext(frame)
        if current_ocr_results:
            last_known_results = []
            for (bbox, text, prob) in current_ocr_results:
                if prob < 0.4 or len(text.strip()) < 2:
                    continue
                if text not in translation_cache:
                    try:
                        translated = translator.translate(text)
                        translated = convert_numerals_to_arabic(translated)
                        translation_cache[text] = translated
                    except:
                        translation_cache[text] = text
                last_known_results.append((bbox, translation_cache[text]))

        if last_known_results:
            inpainted = frame
            for box, _ in last_known_results:
                inpainted = inpaint_text_area(inpainted, box)
            frame = draw_translated_text(inpainted, last_known_results, target_lang)

        out.write(frame)
        progress_data["processed"] = frame_no
        if frame_no % 10 == 0:
            progress_data["message"] = f"Processed {frame_no}/{frame_count} frames..."
            print(progress_data["message"])

    cap.release()
    out.release()
    progress_data.update({"status": "done", "message": "✅ Translation complete!"})

# ================== API Routes ==================
@app.route("/translate", methods=["POST"])
def translate_video():
    file = request.files["file"]
    src_lang = request.form.get("src_lang", "auto")
    target_lang = request.form.get("target_lang", "en")

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    file.save(tmp_in.name)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    thread = threading.Thread(target=process_video, args=(tmp_in.name, tmp_out.name, src_lang, target_lang))
    thread.start()

    return jsonify({"message": "Processing started", "download_path": tmp_out.name})

@app.route("/progress")
def progress_stream():
    def generate():
        last_sent = ""
        while True:
            time.sleep(1)
            json_data = json.dumps(progress_data)
            data = f"data: {json_data}\n\n"
            if data != last_sent:
                yield data
                last_sent = data
            if progress_data["status"] == "done":
                yield f"data: {json_data}\n\n"
                break

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response

@app.route("/download", methods=["GET"])
def download_video():
    path = request.args.get("path")
    return send_file(path, as_attachment=True, download_name="translated_video.mp4")

# ================== Run Flask ==================
if __name__ == "__main__":
    # Optional: set host="0.0.0.0" to access externally
    app.run(port=5000, debug=True)
