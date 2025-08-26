
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def load_parser_model():
    """Loads the pre-trained DeepLabV3 model."""
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def parse_cloth_from_image(model, image_path):
    """
    Takes an image and returns a segmentation mask highlighting the person/clothing.
    """
    input_image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    
    output_predictions = output.argmax(0).cpu().numpy()
    
    # The COCO dataset has 'person' as class 15. We'll use this to find clothing.
    # A model fine-tuned on DeepFashion would be more specific (shirt, pants, etc.).
    person_mask = (output_predictions == 15).astype(np.uint8)
    
    return input_image, person_mask

def visualize_mask(original_image, mask):
    """Displays the original image with the segmentation mask overlaid."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(mask, cmap='jet', alpha=0.5) # Overlay the mask
    plt.title("Parsed Clothing Mask")
    plt.axis('off')
    
    plt.show()

if __name__ == '__main__':
    # NOTE: You'll need a sample image to run this.
    # Create a dummy image or download one into your project folder.
    sample_image_path = 'sample_person.jpg' 
    if not os.path.exists(sample_image_path):
        print(f"'{sample_image_path}' not found. Please provide a sample image of a person to run the demo.")
    else:
        # Load the model
        segmentation_model = load_parser_model()
        
        # Parse the image
        original_img, cloth_mask = parse_cloth_from_image(segmentation_model, sample_image_path)
        
        # Visualize the result
        visualize_mask(original_img, cloth_mask)
