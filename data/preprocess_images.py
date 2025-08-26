
import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

def create_master_annotation_file(raw_data_dir, output_dir):
    """
    Parses the individual DeepFashion annotation files and combines them
    into a single, clean CSV file.
    """
    print("Parsing annotation files...")
    
    # Paths to annotation files
    cat_img_path = os.path.join(raw_data_dir, 'list_category_img.txt')
    cat_cloth_path = os.path.join(raw_data_dir, 'list_category_cloth.txt')
    bbox_path = os.path.join(raw_data_dir, 'list_bbox.txt')

    # Read the category-image mapping
    img_cat_df = pd.read_csv(cat_img_path, sep='\s+', skiprows=2)

    # Read the category names
    cat_names_df = pd.read_csv(cat_cloth_path, sep='\s+', skiprows=2)
    cat_names_map = dict(zip(cat_names_df['category_name'], cat_names_df.index))
    # Invert map for easy lookup
    cat_id_to_name_map = {v: k for k, v in cat_names_map.items()}

    # Read bounding box information
    bbox_df = pd.read_csv(bbox_path, sep='\s+', skiprows=2)
    
    # Merge the dataframes
    master_df = pd.merge(img_cat_df, bbox_df, on='image_name')
    master_df['category_name'] = master_df['category_label'].map(cat_id_to_name_map)
    
    output_path = os.path.join(output_dir, 'master_annotations.csv')
    master_df.to_csv(output_path, index=False)
    
    print(f"Master annotation file created at {output_path}")
    return master_df

def resize_images(raw_data_dir, output_dir, size=(256, 256)):
    """
    Resizes all images in the raw data directory to a standard size.
    """
    print(f"Resizing images to {size}...")
    
    image_folder = os.path.join(raw_data_dir, 'img')
    if not os.path.isdir(image_folder):
        print(f"Error: 'img' folder not found in '{raw_data_dir}'. Please run get_fashion_data.sh first.")
        return

    # Ensure the output directory for images exists
    resized_img_dir = os.path.join(output_dir, 'images_resized')
    os.makedirs(resized_img_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    for filename in tqdm(image_files, desc="Resizing Images"):
        try:
            img_path = os.path.join(image_folder, filename)
            with Image.open(img_path) as img:
                # Resize and convert to RGB to handle various image modes
                img = img.resize(size, Image.Resampling.LANCZOS).convert('RGB')
                img.save(os.path.join(resized_img_dir, filename))
        except Exception as e:
            print(f"Could not process {filename}: {e}")
            
    print("Image resizing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess raw DeepFashion image data.")
    
    parser.add_argument('--input_dir', type=str, default='raw_images',
                        help='Directory containing the raw image folder and annotation files.')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save the resized images and master annotation file.')
    parser.add_argument('--size', type=int, default=256,
                        help='The size (width and height) to resize images to.')
                        
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_master_annotation_file(args.input_dir, args.output_dir)
    resize_images(args.input_dir, args.output_dir, size=(args.size, args.size))
