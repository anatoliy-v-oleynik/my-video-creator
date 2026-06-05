import os
import json
import subprocess
import shutil

print("📥 Начинаем скачивание собранных весов HunyuanVideo 1.5 I2V...")
MODEL_DIR = "/kaggle/working/hunyuan15-fast"
os.makedirs(MODEL_DIR, exist_ok=True)

subprocess.run(["pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import snapshot_download

print("⏳ Скачивание оригинальных весов Tencent в сжатом fp8 формате...")
snapshot_download(
    repo_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
    local_dir=MODEL_DIR,
    max_workers=4,
    local_files_only=False
)
print("✅ Все файлы модели успешно скачаны!")

metadata = {
  "title": "hunyuan15-fast",
  "id": "avonosu/hunyuan15-fast",
  "licenses": [{"name": "CC0-1.0"}]
}
with open(f"{MODEL_DIR}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("🔐 Авторизация утилиты Kaggle CLI...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
except Exception as e:
    print("⚠️ Ошибка секретов:", e)

print("📤 Публикация датасета в ваш профиль Kaggle...")
status = subprocess.run(['kaggle', 'datasets', 'status', 'avonosu/hunyuan15-fast'], capture_output=True, text=True)

if "not found" in status.stderr.lower() or status.returncode != 0:
    print("Создаем новый приватный датасет...")
    subprocess.run(['kaggle', 'datasets', 'create', '-p', MODEL_DIR, '-r']) 
else:
    print("Обновляем существующий датасет...")
    subprocess.run(['kaggle', 'datasets', 'version', '-p', MODEL_DIR, '-m', "Update weights"])

print("🎉 Все процессы завершены!")
