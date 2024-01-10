from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def resize_image(image, target_size):
    return image.resize(target_size)

def convert_to_grayscale(image):
    return image.convert('L')

def calculate_l1(image1, image2):
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    return np.sum(np.abs(arr1 - arr2))

def calculate_mse(image1, image2):
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    return np.mean((arr1 - arr2) ** 2)

def process_images(image_folder):
    results = []

    target_size = (256, 256)

    l1_results_gan = []
    mse_results_gan = []
    l1_results_sd_base = []
    mse_results_sd_base = []
    l1_results_sd_lora = []
    mse_results_sd_lora = []

    for i in range(1, 11):
        filename1 = f"{i}.jpg"
        filename2 = f"{i}_gan.jpg"
        filename3 = f"{i}_sd_base.png"
        filename_lora = f"{i}_sd_lora.png"

        image1 = Image.open(os.path.join(image_folder, filename1))
        image2 = Image.open(os.path.join(image_folder, filename2))
        image3 = Image.open(os.path.join(image_folder, filename3))
        image_lora = Image.open(os.path.join(image_folder, filename_lora))

        # 调整图像尺寸为相同大小
        image1_resized = resize_image(image1, target_size)
        image2_resized = resize_image(image2, target_size)
        image3_resized = resize_image(image3, target_size)
        image_lora_resized = resize_image(image_lora, target_size)

        # 将图像转换为灰度图像
        image1_gray = convert_to_grayscale(image1_resized)
        image2_gray = convert_to_grayscale(image2_resized)
        image3_gray = convert_to_grayscale(image3_resized)
        image_lora_gray = convert_to_grayscale(image_lora_resized)

        l1_diff_gan = calculate_l1(image1_gray, image2_gray)
        mse_gan = calculate_mse(image1_gray, image2_gray)

        l1_diff_sd_base = calculate_l1(image1_gray, image3_gray)
        mse_sd_base = calculate_mse(image1_gray, image3_gray)

        l1_diff_sd_lora = calculate_l1(image1_gray, image_lora_gray)
        mse_sd_lora = calculate_mse(image1_gray, image_lora_gray)

        results.append({
            'Image Pair': f"{filename1} - {filename2}",
            'L1 Difference': l1_diff_gan,
            'MSE': mse_gan
        })

        results.append({
            'Image Pair': f"{filename1} - {filename3}",
            'L1 Difference': l1_diff_sd_base,
            'MSE': mse_sd_base
        })

        results.append({
            'Image Pair': f"{filename1} - {filename_lora}",
            'L1 Difference': l1_diff_sd_lora,
            'MSE': mse_sd_lora
        })

        l1_results_gan.append(l1_diff_gan)
        mse_results_gan.append(mse_gan)
        l1_results_sd_base.append(l1_diff_sd_base)
        mse_results_sd_base.append(mse_sd_base)
        l1_results_sd_lora.append(l1_diff_sd_lora)
        mse_results_sd_lora.append(mse_sd_lora)

    return results, l1_results_gan, mse_results_gan, l1_results_sd_base, mse_results_sd_base, l1_results_sd_lora, mse_results_sd_lora

def save_results_to_csv(results, csv_filename):
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)

def plot_results(l1_results_gan, mse_results_gan, l1_results_sd_base, mse_results_sd_base, l1_results_sd_lora, mse_results_sd_lora):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(l1_results_gan, label='L1 - x.jpg to x_gan.jpg')
    plt.plot(l1_results_sd_base, label='L1 - x.jpg to x_sd_base.jpg')
    plt.plot(l1_results_sd_lora, label='L1 - x.jpg to x_sd_lora.png')
    plt.title('L1 Difference Comparison')
    plt.xlabel('Image Pair')
    plt.ylabel('Difference Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mse_results_gan, label='MSE - x.jpg to x_gan.jpg')
    plt.plot(mse_results_sd_base, label='MSE - x.jpg to x_sd_base.jpg')
    plt.plot(mse_results_sd_lora, label='MSE - x.jpg to x_sd_lora.png')
    plt.title('MSE Comparison')
    plt.xlabel('Image Pair')
    plt.ylabel('Difference Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    image_folder = "duibi"
    csv_filename = "image_diff_results.csv"

    results, l1_results_gan, mse_results_gan, l1_results_sd_base, mse_results_sd_base, l1_results_sd_lora, mse_results_sd_lora = process_images(image_folder)

    save_results_to_csv(results, csv_filename)

    plot_results(l1_results_gan, mse_results_gan, l1_results_sd_base, mse_results_sd_base, l1_results_sd_lora, mse_results_sd_lora)

if __name__ == "__main__":
    main()
