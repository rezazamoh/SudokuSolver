import struct
import numpy as np

def read_hoda_cdb(file_name):
    with open(file_name, 'rb') as f:
        data = f.read()

    images = []
    labels = []
    offset = 0
    file_size = len(data)
    
    while offset < file_size:
        try:
            label = data[offset]
            width = data[offset + 1]
            height = data[offset + 2]
            offset += 5 # Skip reserved bytes
            
            image_size = width * height
            img_data = data[offset:offset + image_size]
            offset += image_size
            
            img_np = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width))
            # Inverse to get black digit on white background
            img_np = np.where(img_np > 0, 0, 255).astype(np.uint8)
            
            images.append(img_np)
            labels.append(label)
        except:
            break
    return images, labels
