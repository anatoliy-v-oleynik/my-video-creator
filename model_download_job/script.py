import os
import json
import subprocess
import shutil
import sys
from pathlib import Path

print("=" * 70)
print("📦 УНИВЕРСАЛЬНЫЙ ЗАГРУЗЧИК МОДЕЛЕЙ ДЛЯ KAGGLE")
print("=" * 70)

# Загружаем конфиг
with open("/kaggle/input/model-downloader/models_config.json", "r") as f:
    config = json.load(f)

print(f"\n📋 Найдено моделей в конфиге: {len(config['models'])}")
for model in config['models']:
    status = "✅ ВКЛЮЧЕНА" if model.get('enabled', False) else "⏸️ ОТКЛЮЧЕНА"
    print(f"   {status} - {model['display_name']} ({model['name']})")

# Устанавливаем общие зависимости
print("\n📦 Устанавливаем общие зависимости...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "kagglehub"])

from huggingface_hub import snapshot_download

# Функция для загрузки Hugging Face модели
def download_hf_model(model_config):
    print(f"\n🎬 Загрузка HF модели: {model_config['display_name']}")
    print(f"   ID: {model_config['repo_id']}")
    print(f"   Датасет: {model_config['dataset_id']}")
    
    model_dir = f"/kaggle/working/{model_config['name']}"
    os.makedirs(model_dir, exist_ok=True)
    
    ignore = model_config.get('ignore_patterns', [])
    
    snapshot_download(
        repo_id=model_config['repo_id'],
        local_dir=model_dir,
        max_workers=4,
        local_files_only=False,
        ignore_patterns=ignore
    )
    
    return model_dir

# Функция для загрузки pip пакета
def download_pip_model(model_config):
    print(f"\n📦 Установка pip пакета: {model_config['display_name']}")
    print(f"   Пакет: {model_config['package_name']}")
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", model_config['package_name']])
    
    # Сохраняем информацию об установленном пакете
    model_dir = f"/kaggle/working/{model_config['name']}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/package_info.txt", "w") as f:
        f.write(f"package: {model_config['package_name']}\n")
        f.write(f"installed: {subprocess.check_output([sys.executable, '-m', 'pip', 'show', model_config['package_name']]).decode()}")
    
    return model_dir

# Функция для загрузки GitHub репозитория
def download_github_model(model_config):
    print(f"\n🐙 Клонирование GitHub: {model_config['display_name']}")
    print(f"   URL: {model_config['repo_url']}")
    
    model_dir = f"/kaggle/working/{model_config['name']}"
    
    subprocess.run(['git', 'clone', model_config['repo_url'], model_dir], check=True)
    
    # Устанавливаем зависимости если есть
    if os.path.exists(f"{model_dir}/requirements.txt"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", f"{model_dir}/requirements.txt"])
    
    return model_dir

# Обработка каждой включённой модели
downloaded_models = []

for model in config['models']:
    if not model.get('enabled', False):
        continue
    
    try:
        if model['repo_type'] == 'huggingface':
            model_dir = download_hf_model(model)
        elif model['repo_type'] == 'pip':
            model_dir = download_pip_model(model)
        elif model['repo_type'] == 'github':
            model_dir = download_github_model(model)
        else:
            print(f"⚠️ Неизвестный тип: {model['repo_type']}")
            continue
        
        # Создаём metadata для датасета
        metadata = {
            "title": model['name'],
            "id": model['dataset_id'],
            "licenses": [{"name": "Apache-2.0"}],
            "description": model['description']
        }
        
        with open(f"{model_dir}/dataset-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        downloaded_models.append({
            "name": model['name'],
            "dataset_id": model['dataset_id'],
            "path": model_dir
        })
        
        print(f"✅ Модель {model['name']} успешно скачана!")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке {model['name']}: {e}")

# Настройка Kaggle API
print("\n🔐 Настройка Kaggle API...")
os.makedirs("/root/.kaggle", exist_ok=True)
try:
    shutil.copy("/kaggle/input/kaggle-api-secret/kaggle.json", "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print("✅ Авторизация настроена")
except Exception as e:
    print(f"⚠️ Ошибка: {e}")

# Загрузка каждого датасета в Kaggle
for model_info in downloaded_models:
    print(f"\n📤 Загрузка {model_info['name']} в Kaggle...")
    os.chdir(model_info['path'])
    
    status = subprocess.run(['kaggle', 'datasets', 'status', model_info['dataset_id']], 
                           capture_output=True, text=True)
    
    if "404" in status.stderr or "not found" in status.stderr.lower():
        print(f"   Создаём новый датасет: {model_info['dataset_id']}")
        subprocess.run(['kaggle', 'datasets', 'create', '-p', model_info['path'], '-r', 'private'])
    else:
        print(f"   Обновляем датасет: {model_info['dataset_id']}")
        subprocess.run(['kaggle', 'datasets', 'version', '-p', model_info['path'], '-m', 'Update model'])

print("\n" + "=" * 70)
print("🎉 ВСЕ МОДЕЛИ ЗАГРУЖЕНЫ!")
print("=" * 70)
print("\n📋 Список доступных датасетов:")
for model_info in downloaded_models:
    print(f"   - {model_info['dataset_id']}")
