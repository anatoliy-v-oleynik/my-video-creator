import os
import json
import subprocess
import time
import shutil
from datetime import datetime
import torch
import imageio
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline

# Эту строчку GitHub Actions будет автоматически перезаписывать вашим промптом
PROMPT = "Космический кот летит на ракете, 4k, high quality"
print(f"🎯 Генерируем по промпту: {PROMPT}")

# 1. Генерируем первый кадр
print("🖼️ Генерируем первый кадр...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = sd_pipe(PROMPT).images[0]
image.save("/kaggle/working/first_frame.png")
print("✅ Первый кадр сохранён")

# Освобождаем VRAM видеокарты, чтобы Kaggle не упал по памяти
del sd_pipe
import gc
gc.collect()
torch.cuda.empty_cache()

# 2. Загружаем SVD модель для анимации
print("🎬 Загружаем модель анимации...")
model_id = "stabilityai/stable-video-diffusion-img2vid"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()

# 3. Генерируем видео
print("🎬 Генерируем видео из картинки...")
frames = pipe(image, decode_chunk_size=8).frames[0]

output_video = "/kaggle/working/output.mp4"
imageio.mimsave(output_video, frames, fps=7)
print(f"✅ Видео сохранено: {output_video}")

# 4. Подготовка структуры датасета
output_dir = "/kaggle/working/output_dataset"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_name = f"generated_{timestamp}.mp4"
shutil.copy(output_video, f"{output_dir}/{video_name}")

metadata = {
    "title": "generated-videos",
    "id": "avonosu/generated-videos",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(f"{output_dir}/dataset-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 5. Авторизация и загрузка в Kaggle Datasets
print("🔐 Подготовка авторизации для Kaggle CLI...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    # Пытаемся взять системный ключ Kaggle
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
except Exception as e:
    print("⚠️ Не удалось скопировать системный токен, используем локальный режим.")

print("📤 Отправка данных в Kaggle...")
status = subprocess.run(['kaggle', 'datasets', 'status', 'avonosu/generated-videos'], capture_output=True, text=True)

if "not found" in status.stderr.lower() or status.returncode != 0:
    print("Инициализация нового датасета...")
    subprocess.run(['kaggle', 'datasets', 'create', '-p', output_dir])
else:
    print("Создание новой версии датасета...")
    subprocess.run(['kaggle', 'datasets', 'version', '-p', output_dir, '-m', f"Prompt: {PROMPT[:50]}"])

print("🎉 Процесс полностью завершен!")
