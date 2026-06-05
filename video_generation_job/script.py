import os
import json
import subprocess
import time
import shutil
from datetime import datetime
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video

# 1. Настройка промпта (перезаписывается через GitHub Actions)
# ЭТИ ДВЕ СТРОЧКИ GITHUB ACTIONS БУДЕТ АВТОМАТИЧЕСКИ ПЕРЕЗАПИСЫВАТЬ ПРИ КАЖДОМ НАЖАТИИ КНОПКИ!
PROMPT = "The wind picks up, the hair starts to flutter softly, cinematically"
print(f"🎯 Сценарий анимации: {PROMPT}")
IMAGE_URL = "https://githubusercontent.com"
print(f"🎯 Адрес изображения: {IMAGE_URL}")

# 2. Загрузка стартового изображения
# Скрипт ищет файл input_image.png, который вы закинули в репозиторий
IMAGE_PATH = "/kaggle/working/input_image.png"

try:
    # Скачиваем файл по ссылке и сохраняем на сверхбыстрый диск Kaggle
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)
    print("📥 Картинка успешно скачана в облако!")
    image = load_image(IMAGE_PATH)
except Exception as e:
    print(f"❌ Не удалось скачать картинку по ссылке, использую заглушку. Ошибка: {e}")
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (720, 480), color = (73, 109, 137))
    d = ImageDraw.Draw(image)
    d.text((10,10), "Error Image", fill=(255,255,0))

print("🖼️ Стартовое изображение готово к анимации!")

# Загружаем модель HunyuanVideo 1.5 480p I2V напрямую из подключенных моделей Kaggle
print("🎬 Загрузка HunyuanVideo 1.5 480p из локального репозитория...")
LOCAL_MODEL_PATH = "/kaggle/input/HunyuanVideo-1.5-Diffusers-480p_i2v"

pipe = DiffusionPipeline.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True # Работаем полностью локально, без интернета!
)

# ВКЛЮЧАЕМ МАКСИМАЛЬНЫЙ ТЮНИНГ ДЛЯ БЕСПЛАТНОЙ КАРТЫ KAGGLE T4
pipe.enable_model_cpu_offload()  # Сбрасывает неиспользуемые блоки в ОЗУ хоста
pipe.vae.enable_tiling()         # Режет видео на тайлы при декодировании, спасая от "Out of Memory"
print("✅ Нейросеть скомпилирована в памяти GPU")

# 4. Запуск генерации видеоролика
print("🚀 Запуск рендеринга... На карте T4 это займет несколько минут.")
# Так как модель distilled, нам нужно ВСЕГО 8-12 шагов вместо стандартных 50!
output = pipe(
    image=image, 
    prompt=PROMPT,
    num_inference_steps=8,       # Скоростной рендеринг
    fps=24                       # Частота кадров плавного видео
).frames[0]

output_video = "/kaggle/working/output_hunyuan.mp4"
export_to_video(output, output_video)
print(f"✅ Видео успешно сгенерировано: {output_video}")

# 5. Сборка датасета для отправки на GitHub / Kaggle
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

# 6. Авторизация утилиты Kaggle CLI и пуш результата
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

print("🎉 Все процессы завершены. Проверьте ваш Kaggle Dataset!")
