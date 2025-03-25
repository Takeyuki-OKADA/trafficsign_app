import sys
import subprocess
import os
import cv2
import numpy as np
import logging
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
import re
from datetime import datetime

# OpenCV のインストールチェック
try:
    import cv2
except ModuleNotFoundError:
    print("OpenCV is missing. Attempting to install...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"])
    import cv2
    print("OpenCV installed successfully:", cv2.__version__)

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# クラスラベル（GTSRB 公式ラベルを使用）
CLASS_LABELS = [
    "制限速度 20km/h", "制限速度 30km/h", "制限速度 50km/h", "制限速度 60km/h", "制限速度 70km/h",
    "制限速度 80km/h", "制限速度 80km/h 終了", "制限速度 100km/h", "制限速度 120km/h", "追い越し禁止",
    "大型追い越し禁止", "次の交差点優先", "優先", "譲れ", "停止", "車両進入禁止", "大型禁止", "進入禁止", "警告", "左カーブ",
    "右カーブ", "連続カーブ", "凹凸", "スリップ", "幅員減少", "工事", "信号", "歩行者", "飛び出し", "自転車",
    "凍結", "動物", "解除", "右折のみ", "左折のみ", "直進", "直進・右折", "直進・左折", "右折専用レーン", "左折専用レーン",
    "環状交差点", "追い越し制限解除", "大型車追い越し制限解除"
]

# 必要なフォルダを作成
for folder in ["debug_images", "input_images", "static", "templates"]:
    os.makedirs(folder, exist_ok=True)

# Flask アプリのセットアップ
app = Flask(__name__)

# 学習済みモデルのロード
model = load_model("./model_R32.keras", compile=False)
logger.info("モデルロード完了")

# ファイル名の正規化
def clean_filename(filename):
    filename = re.sub(r"[^\w\d.]", "_", filename)  # 記号をアンダースコアに置換
    return filename.lower()  # 小文字化

# 画像の前処理（BGRのままリサイズ & 正規化）
def preprocess_image(image_path, save_debug=False):
    img = cv2.imread(image_path)  # BGR で読み込む
    
    if img is None:
        logger.error(f"画像の読み込みに失敗: {image_path}")
        return None
    
    # リサイズ
    resized_img = cv2.resize(img, (32, 32))

    # デバッグ用画像保存
    if save_debug:
        debug_path = f"debug_images/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(debug_path, resized_img)
        logger.info(f"モデルに渡した画像を保存: {debug_path}")

    # 正規化
    img_array = resized_img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # バッチ次元追加
    
    return img_array

# ルートパス（画像アップロード & 推論）
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html", answer="", processing=False)

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", answer="ファイルがありません", processing=False)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", answer="ファイルが選択されていません", processing=False)

        # ファイルを保存
        filename = f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join("input_images", filename)
        file.save(file_path)
        logger.info(f"画像を保存: {file_path}")

        # 画像の前処理
        img_array = preprocess_image(file_path, save_debug=True)
        if img_array is None:
            return render_template("index.html", answer="画像の処理に失敗しました", processing=False)

        try:
            # モデルで推論（最も確率の高いクラスのみ取得）
            predictions = model.predict(img_array)[0]
            predicted_class = np.argmax(predictions)
            answer = f"これは **{CLASS_LABELS[predicted_class]}** です"
            
            logger.info(f"推論結果: {answer.replace('**', '')}")
            return render_template("index.html", answer=answer, processing=False)

        except Exception as e:
            logger.error(f"推論エラー: {e}")
            return render_template("index.html", answer="推論に失敗しました", processing=False)

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("debug_images", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), threaded=True)

