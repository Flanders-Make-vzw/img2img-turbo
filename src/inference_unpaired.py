import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from tqdm import tqdm  # Import tqdm for the progress bar

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='path to the input directory containing images')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    args = parser.parse_args()

    # Validate model_name and model_path
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')
    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')
    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # Initialize the model
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()

    T_val = build_transform(args.image_prep)

    # List all images in the input directory
    images = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]

    # Process each image with a progress bar
    for img_path in tqdm(images, desc="Processing"):
        input_image = Image.open(img_path).convert('RGB')
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
            output = model(x_t, direction=args.direction, caption=args.prompt)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        # Save the output image
        bname = os.path.basename(img_path)
        os.makedirs(args.output_dir, exist_ok=True)
        output_pil.save(os.path.join(args.output_dir, bname))