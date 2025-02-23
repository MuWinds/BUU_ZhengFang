import os
import numpy as np
from PIL import Image
import pickle
from numba import njit
from concurrent.futures import ThreadPoolExecutor
import time

def stay_blue2gray(image):
    image = image.convert('RGB')
    img_array = np.array(image)
    # Optimized masking using NumPy vectorized operations
    # Relaxed blue mask to preserve more details
    blue_mask = (img_array[:, :, 0] <= 50) & (img_array[:, :, 1] <= 50) & (img_array[:, :, 2] >= 65)
    img_array[blue_mask] = [0, 0, 0]
    img_array[~blue_mask] = [255, 255, 255]
    return Image.fromarray(np.uint8(img_array)).convert('L')

def split_image(image):
    images = []
    x, y, w, h = 5, 0, 12, 23
    for _ in range(4): # changed to _ since i is not used
        images.append(image.crop((x, y, x + w, y + h)))
        x += w
    return images

@njit(nogil=True, cache=True)
def calculate_diff(image_array, model):
    diff = np.bitwise_xor(image_array, model)
    return np.sum(diff)

def load_models(dir_now):
    models = []
    file_names = []
    model_path = os.path.join(dir_now, 'zfgetcode/data/model')
    for filename in os.listdir(model_path):
        model = Image.open(os.path.join(model_path, filename)).convert('L')
        file_names.append(filename[0:1])
        # Pre-binarize models here and convert to uint8 for memory efficiency
        models.append((np.array(model) > 128).astype(np.uint8))
    return models, file_names

def load_models_cached(dir_now, cache_file="models_cache.pkl"):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    models, file_names = load_models(dir_now)
    with open(cache_file, 'wb') as f:
        pickle.dump((models, file_names), f)
    return models, file_names


def single_char_ocr(image, models, file_names):
    image_array = (np.array(image) > 128).astype(np.uint8)  # 二值化字符
    min_count = float('inf')
    best_match = None
    for i, model in enumerate(models):
        if model.shape[1] != image_array.shape[1]:
            continue
        model_bin = (model > 128).astype(np.uint8)  # 二值化模板
        count = calculate_diff(image_array, model_bin)
        if count < min_count:
            min_count = count
            best_match = file_names[i]
    return best_match if best_match else "#"

def ocr(images, models, file_names):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(single_char_ocr, images, [models]*len(images), [file_names]*len(images)))
    return "".join(results)

def run(image_path, dir_now):
    run_before = time.time()
    image = Image.open(os.path.join(image_path, "code.jpg"))
    image = stay_blue2gray(image)
    images = split_image(image)
    models, file_names = load_models_cached(dir_now)
    result = ocr(images, models, file_names)
    print(result)
    print("耗时：",time.time()-run_before,"s")
    return result

if __name__ == "__main__":
    result = run(os.getcwd(), os.getcwd())
    print(result)
