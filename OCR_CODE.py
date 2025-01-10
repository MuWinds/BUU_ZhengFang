import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def stay_blue2gray(image):
    image = image.convert('RGB')
    img_array = np.array(image)
    
    # 优化的蓝色过滤，直接对所有像素进行批量操作
    mask = (img_array[:, :, 0] <= 40) & (img_array[:, :, 1] <= 40) & (img_array[:, :, 2] >= 65)
    img_array[mask] = [0, 0, 0]
    img_array[~mask] = [255, 255, 255]
    
    return Image.fromarray(np.uint8(img_array)).convert('L')


def split_image(image):
    images = []
    x, y, w, h = 5, 0, 12, 23
    for i in range(4):
        images.append(image.crop((x, y, x + w, y + h)))
        x += w
    return images


def load_models(dir_now):
    models = []
    file_names = []
    
    model_path = os.path.join(dir_now, 'zfgetcode/data/model')
    for filename in os.listdir(model_path):
        model = Image.open(os.path.join(model_path, filename)).convert('L')
        file_names.append(filename[0:1])
        models.append(np.array(model))
    
    return models, file_names


def single_char_ocr(image, models, file_names):
    result = "#"
    image_array = np.array(image)

    min_count = image.size[0] * image.size[1]
    best_match = None

    for i, model in enumerate(models):
        if model.shape[1] != image_array.shape[1]:
            continue

        # 计算差异度，使用 NumPy 高效计算
        diff = np.abs(image_array - model)
        count = np.sum(diff > 0)

        if count < min_count:
            min_count = count
            best_match = file_names[i]

    return best_match if best_match else result


def ocr(images, models, file_names):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda image: single_char_ocr(image, models, file_names), images))
    return "".join(results)

def run(image_path, dir_now):
    image = Image.open(os.path.join(image_path, "code.jpg"))
    image = stay_blue2gray(image)
    images = split_image(image)
    models, file_names = load_models(dir_now)
    result = ocr(images, models, file_names)
    return result


if __name__ == "__main__":
    result = run(os.getcwd(), os.getcwd())
    print(result)
