from PIL import Image
import os

os.makedirs('data/images', exist_ok=True)
colors = [(40, 255, 100), (80, 95, 150), (120, 15, 200)]
for i, color in enumerate(colors):
    img = Image.new('RGB', (100, 100), color)
    img.save(f'data/images/img{i+1}.jpg') 