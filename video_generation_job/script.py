import os
import json
import subprocess
import shutil
import urllib.request
import sys
from datetime import datetime
import torch

print("=" * 70)
print("🎬 ГЕНЕРАЦИЯ ВИДЕО HUNYUANVIDEO 1.5")
print("=" * 70)

# ==========================================================
# НАСТРОЙКИ (меняйте здесь при необходимости)
# ==========================================================
PROMPT = "The background starts moving beautifully, high quality"
IMAGE_URL = "https://github.com/anatoliy-v-oleynik/my-video-creator/blob/main/video_generation_job/input_image.png?raw=true"

# Путь к модели в датасете Kaggle (уже загружена, не качается!)
MODEL_PATH = "/kaggle/input/hunyuanvideo-1.5-diffusers-480p-i2v"
# ==========================================================

print(f"🎯 Промпт: {PROMPT}")
print(f"📸 URL картинки: {IMAGE_URL}")
print(f"📦 Путь к модели: {MODEL_PATH}")

# Проверяем, есть ли модель в датасете
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ОШИБКА: Модель не найдена по пути {MODEL_PATH}")
    print("📌 Убедитесь, что в kernel-metadata.json добавлен dataset_sources:")
    print('   "dataset_sources": ["hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"]')
    sys.exit(1)

print(f"✅ Модель найдена в датасете Kaggle!")
print(f"📁 Содержимое: {os.listdir(MODEL_PATH)[:5]}...")

# Устанавливаем accelerate (нужен для cpu_offload)
print("\n📦 Устанавливаем accelerate...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "accelerate"])

from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

# ==========================================================
# ЗАГРУЗКА МОДЕЛИ (из датасета, мгновенно!)
# ==========================================================
print("\n🎬 Загрузка модели из Kaggle Dataset...")
pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True  # КЛЮЧЕВОЙ ПАРАМЕТР — не выходим в интернет!
)

# Оптимизация для Kaggle T4 (16GB VRAM)
print("🔄 Оптимизация памяти...")
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
print("✅ Модель загружена и оптимизирована!")

# ==========================================================
# ЗАГРУЗКА КАРТИНКИ
# ==========================================================
print("\n📥 Загрузка стартовой картинки...")
IMAGE_PATH = "/kaggle/working/input_image.png"

try:
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print("✅ Картинка успешно скачана!")
    image = load_image(IMAGE_PATH)
except Exception as e:
    print(f"⚠️ Ошибка загрузки картинки: {e}")
    print("🖼️ Создаём тестовую картинку...")
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (720, 480), color=(73, 109, 137))
    d = ImageDraw.Draw(image)
    d.text((10, 10), "Test Image", fill=(255, 255, 0))

print(f"🖼️ Размер картинки: {image.size}")

# ==========================================================
# ГЕНЕРАЦИЯ ВИДЕО
# ==========================================================
print("\n🚀 СТАРТ ГЕНЕРАЦИИ ВИДЕО...")
print("⏱️ Это займёт 5-10 минут...")
print("🎬 Создаём видео 480p, 89 кадров (3.7 секунды при 24fps)")

generator = torch.Generator(device="cuda:0").manual_seed(42)

video_frames = pipe(
    prompt=PROMPT,
    image=image,
    generator=generator,
    num_frames=89,           # ~3.7 секунды при 24 fps
    num_inference_steps=30,  # Баланс скорость/качество
).frames[0]

output_video = "/kaggle/working/output_hunyuan.mp4"
export_to_video(video_frames, output_video, fps=24)
print(f"✅ ВИДЕО СОЗДАНО: {output_video}")

# ==========================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТА В ДАТАСЕТ
# ==========================================================
print("\n📤 Сохранение видео в датасет...")

output_dir = "/kaggle/working/output_dataset"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = f"hunyuan_{timestamp}.mp4"
output_path = os.path.join(output_dir, video_name)
shutil.copy(output_video, output_path)

# Проверяем размер видео
video_size = os.path.getsize(output_path) / (1024 * 1024)
print(f"📊 Размер видео: {video_size:.2f} MB")

# Создаём metadata для датасета
metadata = {
    "title": "generated-videos",
    "id": "avonosu/generated-videos",
    "licenses": [{"name": "CC0-1.0"}],
    "description": f"Generated with prompt: {PROMPT[:100]}"
}
with open(f"{output_dir}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ==========================================================
# АВТОРИЗАЦИЯ KAGGLE
# ==========================================================
print("\n🔐 Настройка Kaggle API...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✅ Авторизация настроена")
except Exception as e:
    print(f"⚠️ Ошибка авторизации: {e}")

# ==========================================================
# ЗАГРУЗКА В KAGGLE DATASET
# ==========================================================
print(f"\n📤 Загрузка датасета avonosu/generated-videos...")
os.chdir(output_dir)

# Проверяем, существует ли датасет
status = subprocess.run(
    ['kaggle', 'datasets', 'status', 'avonosu/generated-videos'],
    capture_output=True,
    text=True
)

if "404" in status.stderr or "not found" in status.stderr.lower():
    print("📦 Создаём новый датасет...")
    result = subprocess.run(
        ['kaggle', 'datasets', 'create', '-p', output_dir, '-r', 'private'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
else:
    print("🔄 Обновляем существующий датасет...")
    result = subprocess.run(
        ['kaggle', 'datasets', 'version', '-p', output_dir, '-m', f"Video: {PROMPT[:50]}"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

# ==========================================================
# ФИНАЛ
# ==========================================================
print("\n" + "=" * 70)
print("🎉 ВСЕ ГОТОВО!")
print("=" * 70)
print(f"📦 Датасет с видео: avonosu/generated-videos")
print(f"📹 Файл видео: {video_name}")
print(f"🎬 Промпт: {PROMPT}")
print("=" * 70)

# Выводим информацию о времени выполнения
print(f"\n⏱️ Видео создано за ~10 минут")
print("🔗 Ссылка на датасет: https://www.kaggle.com/avonosu/generated-videos")
