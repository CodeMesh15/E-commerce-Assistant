#!/bin/bash

# This script provides instructions for downloading the DeepFashion Dataset
# needed for the AI-Powered E-commerce Assistant project.

# Create the directory for the raw data
mkdir -p raw_images

echo "--- AI E-commerce Assistant: Data Download Guide ---"
echo ""
echo "This script will not download the data automatically due to the dataset's hosting."
echo "Please follow these manual steps:"
echo ""
echo "1. Open a web browser and go to the DeepFashion project page:"
echo "   http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html"
echo ""
echo "2. Navigate to the 'Category and Attribute Prediction Benchmark' section."
echo "3. Click the download link. This will take you to a Google Drive folder."
echo "4. Download the following files:"
echo "   - img.zip (This contains all the images)"
echo "   -Anno/list_bbox.txt"
echo "   -Anno/list_category_cloth.txt"
echo "   -Anno/list_category_img.txt"
echo ""
echo "5. Unzip 'img.zip' and move its contents (the 'img' folder) into the"
echo "   'data/raw_images/' directory created in this project."
echo ""
echo "6. Move the downloaded .txt annotation files into the same 'data/raw_images/' directory."
echo ""
echo "After these files are in place, you can run the preprocess_images.py script."
echo ""
echo "----------------------------------------------------"
