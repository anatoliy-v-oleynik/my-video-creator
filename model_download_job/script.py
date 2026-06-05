import os
import json
import subprocess
import shutil
import sys

print("=" * 70)
print("📦 ЗАГРУЗЧИК МОДЕЛИ В ДАТАСЕТ KAGGLE")
print("=" * 70)

# ==========================================================
# НАСТРОЙКИ (меняйте здесь при необходимости)
# ==========================================================
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
DATASET_NAME = "hunyuan15-480p-model"
DATASET_ID = f"avonosu/{DATASET_NAME}"
# ==========================================================

print(f"🎯 Модель: {MODEL_ID}")
print(f"📦 Датасет: {DATASET_ID}")

# Устанавливаем зависимости
print("\n📦 Устанавливаем зависимости...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import snapshot_download

# Скачиваем модель
print("\n⏳ Скачивание модели... (5-15 минут)")
model_dir = f"/kaggle/working/model"
os.makedirs(model_dir, exist_ok=True)

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=model_dir,
        max_workers=4,
        ignore_patterns=["*.bin", "*.msgpack"]
    )
    print("✅ Модель успешно скачана!")
except Exception as e:
    print(f"❌ Ошибка при скачивании: {e}")
    sys.exit(1)

# Создаём metadata для датасета
print("\n📝 Создаём metadata...")
metadata = {
    "title": DATASET_NAME,
    "id": DATASET_ID,
    "licenses": [{"name": "Apache-2.0"}],
    "description": f"Model: {MODEL_ID}"
}
with open(f"{model_dir}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Настройка Kaggle API
print("\n🔐 Настройка Kaggle API...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✅ Авторизация настроена")
except Exception as e:
    print(f"⚠️ Ошибка: {e}")

# Загрузка датасета в Kaggle
print(f"\n📤 Загрузка датасета {DATASET_ID} в Kaggle...")
os.chdir(model_dir)

# Проверяем, существует ли уже датасет
status = subprocess.run(['kaggle', 'datasets', 'status', DATASET_ID], 
                       capture_output=True, text=True)

if "404" in status.stderr or "not found" in status.stderr.lower():
    print("   Создаём новый датасет...")
    result = subprocess.run(['kaggle', 'datasets', 'create', '-p', model_dir, '-r', 'private'],
                           capture_output=True, text=True)
    print(result.stdout)
else:
    print("   Обновляем существующий датасет...")
    result = subprocess.run(['kaggle', 'datasets', 'version', '-p', model_dir, '-m', 'Update model'],
                           capture_output=True, text=True)
    print(result.stdout)

print("\n" + "=" * 70)
print("🎉 ГОТОВО!")
print(f"📦 Датасет: {DATASET_ID}")
print("=" * 70)
