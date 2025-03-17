import os
import cv2
import numpy as np
import tensorflow as tf
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from datetime import datetime

# ✅ CUDA を無効化（Render は GPU 未対応）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Flask アプリのセットアップ
app = Flask(__name__, static_folder="static", template_folder="templates")

# ✅ `index.html` を表示するエンドポイント (Web UI)
@app.route("/", methods=["GET", "HEAD"])
def home():
    return render_template("index.html", answer="", processing=False)

# ✅ `favicon.ico` の 404 エラーを防ぐ
@app.route('/favicon.ico')
def favicon():
    return send_from_directory("static", "favicon.ico", mimetype="image/vnd.microsoft.icon")

# ✅ `static` フォルダを明示的に提供（CSS, JS, 画像）
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

# ✅ Flask アプリ起動 (Render用)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render の PORT 環境変数を取得
    logger.info(f"🚀 アプリ起動: ポート {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
