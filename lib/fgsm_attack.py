import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils


def gen_perturbation_pic(model, image, target, epsilon):
    # Set the model to evaluation mode
    model.eval()
    # Enable gradient calculation for the input image
    image.requires_grad = True

    # Forward pass
    output = model(image)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()

    # Collect gradients
    data_grad = image.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def gen_perturbation_file(model, test_loader, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, (image, label) in enumerate(test_loader):
        perturbed_image = gen_perturbation_pic(model, image, label, epsilon=0.1)  # Adjust epsilon as needed
        class_label = label.item()
        class_output_folder = os.path.join(output_folder, str(class_label))
        if not os.path.exists(class_output_folder):
            os.makedirs(class_output_folder)
        save_path = os.path.join(class_output_folder, f'image_{idx}.png')
        vutils.save_image(perturbed_image, save_path)
        print(f'Processed {idx+1} images')
