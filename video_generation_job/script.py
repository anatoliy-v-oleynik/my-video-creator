import os
import json
import subprocess
import time
import shutil
import urllib.request  # Добавили обязательный импорт для скачивания
from datetime import datetime
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

# ЭТИ ДВЕ СТРОЧКИ GITHUB ACTIONS БУДЕТ АВТОМАТИЧЕСКИ ПЕРЕЗАПИСЫВАТЬ!
PROMPT = "The background starts moving beautifully, high quality"
IMAGE_URL = "https://github.com/anatoliy-v-oleynik/my-video-creator/blob/main/video_generation_job/input_image.png?raw=true"

print(f"🎯 Сценарий анимации: {PROMPT}")
print(f"🎯 Адрес изображения: {IMAGE_URL}")

IMAGE_PATH = "/kaggle/working/input_image.png"

# Скачивание изображения с GitHub
try:
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print("📥 Картинка успешно скачана по ссылке!")
    image = load_image(IMAGE_PATH)
except Exception as e:
    print(f"❌ Не удалось скачать картинку по ссылке, использую заглушку. Ошибка: {e}")
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (720, 480), color = (73, 109, 137))
    d = ImageDraw.Draw(image)
    d.text((10,10), "Error Image", fill=(255,255,0))

print("🖼️ Стартовое изображение готово к анимации!")

# Загрузка HunyuanVideo 1.5 480p из локального репозитория Kaggle
print("🎬 Загрузка HunyuanVideo 1.5 480p из локального репозитория...")
LOCAL_MODEL_PATH = "/kaggle/input/HunyuanVideo-1.5-Diffusers-480p_i2v"

# Исправлено: вместо device_map="auto" загружаем напрямую в cuda
# Исправлено: вместо torch.bfloat16 используем torch.float16 для совместимости с картой P100
pipe = DiffusionPipeline.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")

# Оптимизация памяти для Kaggle
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
print("✅ Нейросеть успешно скомпилирована в GPU")

# Запуск рендеринга
print("🚀 Запуск рендеринга... Это займет несколько минут.")
output = pipe(
    image=image, 
    prompt=PROMPT,
    num_inference_steps=8,
    fps=24
).frames

output_video = "/kaggle/working/output_hunyuan.mp4"
export_to_video(output, output_video)
print(f"✅ Видео успешно сгенерировано: {output_video}")

# Сборка датасета
output_dir = "/kaggle/working/output_dataset"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = f"hunyuan_{timestamp}.mp4"
shutil.copy(output_video, f"{output_dir}/{video_name}")

metadata = {
  "title": "generated-videos",
  "id": "avonosu/generated-videos",
  "licenses": [{"name": "CC0-1.0"}]
}
with open(f"{output_dir}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Авторизация и пуш в датасеты Kaggle
print("🔐 Авторизация Kaggle...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
except:
    pass

print("📤 Пушим видео в ваш датасет...")
status = subprocess.run(['kaggle', 'datasets', 'status', 'avonosu/generated-videos'], capture_output=True, text=True)

if "not found" in status.stderr.lower() or status.returncode != 0:
    subprocess.run(['kaggle', 'datasets', 'create', '-p', output_dir])
else:
    subprocess.run(['kaggle', 'datasets', 'version', '-p', output_dir, '-m', f"Hunyuan I2V: {PROMPT[:50]}"])

print("🎉 Процесс полностью завершен!")
