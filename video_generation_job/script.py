import os
import json
import subprocess
import shutil
import urllib.request
from datetime import datetime
import torch

# Устанавливаем accelerate (нужен для cpu_offload)
import subprocess
import sys


# ==========================================================
# КРИТИЧЕСКИ ВАЖНО: Установка accelerate ДО загрузки модели
# ==========================================================
print("📦 Устанавливаем accelerate для cpu_offload...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "accelerate"])

from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ЭТИ СТРОЧКИ ПЕРЕЗАПИСЫВАЮТСЯ GITHUB ACTIONS
PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard"
IMAGE_URL = "https://github.com/anatoliy-v-oleynik/my-video-creator/blob/main/video_generation_job/input_image.png?raw=true"

print(f"🎯 Промпт: {PROMPT}")
print(f"📸 URL картинки: {IMAGE_URL}")

# 1. Скачиваем картинку
IMAGE_PATH = "/kaggle/working/input_image.png"
try:
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print("✅ Картинка скачана")
    image = load_image(IMAGE_PATH)
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (720, 480), color=(73, 109, 137))
    d = ImageDraw.Draw(image)
    d.text((10, 10), "Image Load Error", fill=(255, 255, 0))

# 2. Загружаем модель (ОФИЦИАЛЬНЫЙ СПОСОБ)
print("🎬 Загрузка HunyuanVideo 1.5 I2V...")
dtype = torch.bfloat16
device = "cuda:0"

pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
    torch_dtype=dtype
)

# Оптимизация для Kaggle T4 (16GB VRAM)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
print("✅ Модель загружена и оптимизирована")

# 3. Генерация видео
print("🚀 Старт генерации... (5-10 минут)")
generator = torch.Generator(device=device).manual_seed(42)  # Фиксируем seed

video_frames = pipe(
    prompt=PROMPT,
    image=image,
    generator=generator,
    num_frames=89,          # ~3.7 секунды при 24 fps
    num_inference_steps=30, # Баланс скорость/качество (50 — лучше, но дольше)
).frames[0]  # Берём первый (и единственный) batch

output_video = "/kaggle/working/output_hunyuan.mp4"
export_to_video(video_frames, output_video, fps=24)
print(f"✅ Видео сохранено: {output_video}")

# 4. Отправка в Kaggle Dataset
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

# 5. Авторизация и пуш в Kaggle
print("🔐 Авторизация Kaggle...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✅ Авторизация настроена")
except Exception as e:
    print(f"⚠️ Предупреждение: {e}")

print("📤 Загрузка видео в датасет...")
status = subprocess.run(['kaggle', 'datasets', 'status', 'avonosu/generated-videos'], 
                        capture_output=True, text=True)

if "not found" in status.stderr.lower() or status.returncode != 0:
    subprocess.run(['kaggle', 'datasets', 'create', '-p', output_dir])
else:
    subprocess.run(['kaggle', 'datasets', 'version', '-p', output_dir, 
                    '-m', f"Prompt: {PROMPT[:50]}"])

print("🎉 Готово! Видео в датасете avonosu/generated-videos")
