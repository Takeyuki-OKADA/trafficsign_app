import os
import cv2
import numpy as np
import tensorflow as tf
import logging
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from datetime import datetime

# ✅ ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ クラスラベル（GTSRB 公式ラベルを使用）
CLASS_LABELS = [
    "制限速度 20km/h", "制限速度 30km/h", "制限速度 50km/h", "制限速度 60km/h", "制限速度 70km/h",
    "制限速度 80km/h", "制限速度 80km/h 終了", "制限速度 100km/h", "制限速度 120km/h", "追い越し禁止",
    "大型追い越し禁止", "次の交差点優先", "優先", "譲れ", "停止", "車両進入禁止", "大型禁止", "進入禁止", "警告", "左カーブ",
    "右カーブ", "連続カーブ", "凹凸", "スリップ", "幅員減少", "工事", "信号", "歩行者", "飛び出し", "自転車",
    "凍結", "動物", "解除", "右折のみ", "左折のみ", "直進", "直進・右折", "直進・左折", "右折専用レーン", "左折専用レーン",
    "環状交差点", "追い越し制限解除", "大型車追い越し制限解除"
]

# ✅ 必要なフォルダを作成
for folder in ["debug_images", "input_images", "static", "templates"]:
    os.makedirs(folder, exist_ok=True)

# ✅ 最新の学習済みモデルをロード
def get_latest_model():
    model_files = sorted([f for f in os.listdir() if f.startswith("model_R") and f.endswith(".keras")])
    latest_model = model_files[-1] if model_files else "model_R0.keras"
    logger.info(f"✅ 使用モデル: {latest_model}")
    return load_model(latest_model, compile=False)

try:
    model = get_latest_model()
except Exception as e:
    logger.error(f"❌ モデルロード失敗: {e}")
    exit(1)

# ✅ 画像の前処理（BGRのままリサイズ & 正規化）
def preprocess_image(image_path, save_debug=False):
    img = cv2.imread(image_path)  # BGR で読み込む
    
    if img is None:
        logger.error(f"❌ 画像の読み込みに失敗: {image_path}")
        return None
    
    # リサイズ
    resized_img = cv2.resize(img, (32, 32))

    # デバッグ用画像保存（BGRのまま保存）
    if save_debug:
        debug_path = f"debug_images/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(debug_path, resized_img)
        logger.info(f"✅ モデルに渡した画像を保存: {debug_path}")

    # 正規化
    img_array = resized_img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # バッチ次元追加
    
    return img_array

# ✅ Flask アプリのセットアップ
app = Flask(__name__)

# ✅ ルートパス（Web UI）
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("index.html", answer="", processing=False)

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", answer="❌ ファイルがありません", processing=False)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", answer="❌ ファイルが選択されていません", processing=False)

        # ✅ ファイルを保存
        filename = f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join("input_images", filename)
        file.save(file_path)
        logger.info(f"✅ 画像を保存: {file_path}")

        # ✅ 画像の前処理
        img_array = preprocess_image(file_path, save_debug=True)
        if img_array is None:
            return render_template("index.html", answer="❌ 画像の処理に失敗しました", processing=False)

        try:
            # ✅ モデルで推論（最も確率の高いクラスのみ取得）
            predictions = model.predict(img_array)[0]
            predicted_class = np.argmax(predictions)
            answer = f"これは **{CLASS_LABELS[predicted_class]}** です"
            
            logger.info(f"✅ 推論結果: {answer.replace('**', '')}")
            return render_template("index.html", answer=answer, processing=False)

        except Exception as e:
            logger.error(f"❌ 推論エラー: {e}")
            return render_template("index.html", answer="❌ 推論に失敗しました", processing=False)

# ✅ REST API で推論
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "❌ 画像がアップロードされていません"}), 400

    file = request.files["file"]
    filename = f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    file_path = os.path.join("input_images", filename)
    file.save(file_path)
    logger.info(f"✅ 画像を保存: {file_path}")

    # 画像の前処理
    img = preprocess_image(file_path)
    if img is None:
        return jsonify({"error": "❌ 画像の処理に失敗しました"}), 400

    # 推論
    try:
        predictions = model.predict(img)[0]
        predicted_class = int(np.argmax(predictions))
        return jsonify({"prediction": predicted_class, "label": CLASS_LABELS[predicted_class]})
    except Exception as e:
        logger.error(f"❌ 推論エラー: {e}")
        return jsonify({"error": "❌ 推論に失敗しました"}), 500

# ✅ Flask アプリ起動 (Render用)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Renderの環境変数 PORT を使用
    logger.info(f"🚀 アプリ起動: ポート {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
