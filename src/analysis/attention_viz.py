import numpy as np
import pydicom
import torch
import timm
from PIL import Image
from torchvision import transforms
from timm.utils import AttentionExtract

# Disable fused attention for compatibility
import timm.layers
timm.layers.set_fused_attn(False)

# Load pre-trained DINOv2 model (choose your variant)
model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
model.eval()

dcm_img = pydicom.dcmread("/home/vhari/dom_ameen_chi_link/common/SENTINL0/DINOv2/nlst_test_data/100005/01-02-2001-NLST-LSS-18578/3.000000-2OPAGELS16D3702.514060.00.11.375-27113/1-079.dcm")
# Convert DICOM image to PIL format
dcm_image = dcm_img.pixel_array
dcm_image = Image.fromarray(dcm_image.astype(np.float32)).convert('RGB')
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = transform(dcm_image).unsqueeze(0)  # Add batch dimension

# Set up attention extractor
extractor = AttentionExtract(model, method='fx')

# Extract attention maps
with torch.no_grad():
    attention_maps = extractor(img_tensor)

# Print available attention maps (layer names)
print(attention_maps.keys())
# Example output: dict_keys(['blocks.0.attn.softmax', ..., 'blocks.11.attn.softmax'])

# Extract the last layer's attention map
last_layer = sorted(attention_maps.keys())[-1]
attn = attention_maps[last_layer]  # Shape: (batch, heads, tokens, tokens)

# For visualization or further processing, you may want to average over heads:
attn_map = attn.mean(dim=1)  # Shape: (batch, tokens, tokens)

# Reshape attention map for visualization
# Remove batch dimension and get the attention weights for the [CLS] token
attn_map = attn_map[0, 0, 1:]  # Remove [CLS] token attention to itself
# Reshape to 2D grid (assuming square input)
grid_size = int(np.sqrt(attn_map.shape[0]))
attn_map = attn_map.reshape(grid_size, grid_size)

# Convert to numpy and normalize to 0-255 range for visualization
attn_map = attn_map.cpu().numpy()
attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
attn_map = (attn_map * 255).astype(np.uint8)

# Create a heatmap from the attention map
heatmap = Image.fromarray(attn_map)
heatmap.save("outputs/attention_heatmap.png")

# Display the heatmap
# heatmap.show()