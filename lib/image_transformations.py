import os
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms


def gen_transformed_pic(image, image_transforms):
    pil_image = TF.to_pil_image(image)

    transformed_image = None
    if image_transforms == "median_smooth":
        transformed_image = pil_image.filter(ImageFilter.MedianFilter())
    elif "color_depth_reduction_" in image_transforms:
        bit_to_reduce = int(image_transforms.split('_')[-1])
        np_image = np.array(pil_image)
        np_image = np_image - (np_image % (2 ** bit_to_reduce))
        transformed_image = Image.fromarray(np_image)

    elif "spatial_smoothing_" in image_transforms:
        size_of_window = int(image_transforms.split('_')[-1])
        transformed_image = pil_image.filter(ImageFilter.MedianFilter(size=size_of_window))

    elif image_transforms == "non_local_mean":
        transformed_image = pil_image.filter(ImageFilter.MedianFilter())

    elif image_transforms == "sharpen":
        transformed_image = pil_image.filter(ImageFilter.SHARPEN)

    elif image_transforms == "enhance_color":
        enhancer = ImageEnhance.Color(pil_image)
        transformed_image = enhancer.enhance(2)

    elif "enhance_contrast_" in image_transforms:
        enhancement_factor = float(image_transforms.split('_')[-1])
        enhancer = ImageEnhance.Contrast(pil_image)
        transformed_image = enhancer.enhance(enhancement_factor)

    elif image_transforms == "normalized":
        # Normalize the image
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformed_image = TF.to_tensor(pil_image)
        transformed_image = transform(transformed_image)
        return transformed_image.unsqueeze(0)
    elif "bit_quantization_" in image_transforms:
        bit_depth = int(image_transforms.split('_')[-1])
        np_image = np.array(pil_image)
        np_image = (np_image >> bit_depth) << bit_depth
        transformed_image = Image.fromarray(np_image)
    else:
        # If no valid transformation specified, return original image
        transformed_image = pil_image
        print(f"Invalid transformation specified: {image_transforms}")

    # Convert the transformed PIL Image back to torch tensor
    transformed_image = TF.to_tensor(transformed_image)

    # Unsqueeze the transformed tensor to add a batch dimension
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image


def gen_transformed_file(test_loader, output_folder, image_transforms):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, (image, label) in enumerate(test_loader):
        image = image.squeeze(0)
        transformed_image = gen_transformed_pic(image, image_transforms)
        class_label = label.item()
        class_output_folder = os.path.join(output_folder, str(class_label))
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        save_path = os.path.join(class_output_folder, f'image_{idx}.png')
        vutils.save_image(transformed_image, save_path)
        # print(f'Processed {idx + 1} images')
