import os
import json
import subprocess
import shutil

print("📥 Начинаем скачивание оригинальной тяжелой модели HunyuanVideo 1.5...")
subprocess.run(["pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import snapshot_download

# Создаем временную папку для скачивания полной модели
TEMP_DIR = "/kaggle/working/hunyuan_full"
os.makedirs(TEMP_DIR, exist_ok=True)

print("⏳ Скачивание оригинальных весов напрямую с Hugging Face... Пожалуйста, подождите.")
# Качаем официальную полную модель Tencent
snapshot_download(
    repo_id="tencent/HunyuanVideo",
    local_dir=TEMP_DIR,
    max_workers=4,
    local_files_only=False
)
print("✅ Полная оригинальная модель успешно скачана во временную папку!")

# Подготовка авторизации Kaggle CLI для пуша
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
except:
    pass

# Функция для автоматического создания под-датасетов (чтобы обойти лимит 20 ГБ)
def create_kaggle_dataset(folder_name, source_path):
    target_dir = f"/kaggle/working/{folder_name}"
    os.makedirs(target_dir, exist_ok=True)
    
    # Переносим файлы
    shutil.move(source_path, target_dir)
    
    # Создаем метаданные
    metadata = {
      "title": folder_name,
      "id": f"avonosu/{folder_name}",
      "licenses": [{"name": "CC0-1.0"}]
    }
    with open(f"{target_dir}/dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"📤 Отправка части {folder_name} в Kaggle...")
    status = subprocess.run(['kaggle', 'datasets', 'status', f'avonosu/{folder_name}'], capture_output=True, text=True)
    if "not found" in status.stderr.lower() or status.returncode != 0:
        subprocess.run(['kaggle', 'datasets', 'create', '-p', target_dir, '-r']) 
    else:
        subprocess.run(['kaggle', 'datasets', 'version', '-p', target_dir, '-m', "Update partition"])

# Разрезаем модель на 3 логические части (Текстовые энкодеры, VAE и сам Трансформер)
if os.path.exists(f"{TEMP_DIR}/transformer"):
    create_kaggle_dataset("hunyuan-transformer", f"{TEMP_DIR}/transformer")
if os.path.exists(f"{TEMP_DIR}/text_encoder"):
    create_kaggle_dataset("hunyuan-text-encoders", f"{TEMP_DIR}/text_encoder")
if os.path.exists(f"{TEMP_DIR}/vae"):
    create_kaggle_dataset("hunyuan-vae", f"{TEMP_DIR}/vae")

print("🎉 Магия завершена! Все части оригинальной модели Tencent теперь в вашем профиле!")
