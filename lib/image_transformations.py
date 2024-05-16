import os
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from PIL import ImageFilter


def gen_transformed_pic(image, image_transforms):
    # Convert the torch tensor image to PIL Image
    pil_image = TF.to_pil_image(image)

    transformed_image = None

    if image_transforms == "median_smooth":
        # Apply a median smoothing filter
        transformed_image = pil_image.filter(ImageFilter.MedianFilter())
    elif image_transforms == "bit_quantization":
        # Apply a bit quantization
        transformed_image = pil_image.quantize(colors=2 ** 8)
    elif image_transforms == "non_local_mean":
        # Apply non-local mean filtering
        transformed_image = pil_image.filter(ImageFilter.MedianFilter())
    elif image_transforms == "total_variance_minimization":
        # Apply total variance minimization
        transformed_image = pil_image  # Implement your method here
    elif image_transforms == "image_quilting":
        # Apply image quilting
        transformed_image = pil_image  # Implement your method here
    else:
        # If no valid transformation specified, return original image
        transformed_image = pil_image

    # Convert the transformed PIL Image back to torch tensor
    transformed_image = TF.to_tensor(transformed_image)

    # Ensure proper channel dimension (add a singleton dimension at the beginning)
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image


def gen_transformed_file(test_loader, output_folder, image_transforms):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, (image, label) in enumerate(test_loader):
        image = image.squeeze(0)
        perturbed_image = gen_transformed_pic(image, image_transforms)
        class_label = label.item()
        class_output_folder = os.path.join(output_folder, str(class_label))
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        save_path = os.path.join(class_output_folder, f'image_{idx}.png')
        vutils.save_image(perturbed_image, save_path)
        print(f'Processed {idx + 1} images')
