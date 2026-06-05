import os
import json
import subprocess
import shutil

print("📥 Начинаем скачивание модели HunyuanVideo 1.5 480p i2v для сюжетов...")
subprocess.run(["pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import snapshot_download

# Создаем чистую папку для сбора весов модели
MODEL_DIR = "/kaggle/working/hunyuan15-480p-core"
os.makedirs(MODEL_DIR, exist_ok=True)

# Указываем репозиторий с идеальной структурой под diffusers
REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

print("⏳ Скачивание всех компонентов модели напрямую с Hugging Face...")
# snapshot_download сам скачает все папки (vae, transformer, scheduler) без ошибок 404!
snapshot_download(
    repo_id=REPO,
    local_dir=MODEL_DIR,
    max_workers=4, # Качаем на максимальной гигабитной скорости Kaggle
    local_files_only=False
)
print("✅ Все файлы модели (около 6 ГБ) успешно скачаны!")

# Создаём метаданные для вашего личного датасета на Kaggle
metadata = {
  "title": "hunyuan15-480p-core",
  "id": "avonosu/hunyuan15-480p-core",
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
status = subprocess.run(['kaggle', 'datasets', 'status', 'avonosu/hunyuan15-480p-core'], capture_output=True, text=True)

if "not found" in status.stderr.lower() or status.returncode != 0:
    print("Создаем новый приватный датасет...")
    subprocess.run(['kaggle', 'datasets', 'create', '-p', MODEL_DIR, '-r']) 
else:
    print("Обновляем существующий датасет...")
    subprocess.run(['kaggle', 'datasets', 'version', '-p', MODEL_DIR, '-m', "Update weights"])

print("🎉 Процесс полностью завершен! Модель готова к работе.")
