from fastai.vision.all import *
import matplotlib.pyplot as plt
from PIL import Image

# 📥 Define dataset path (used during training)
path = '/root/.cache/kagglehub/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5/animals/animals'

# 🏗 Recreate the DataLoaders (needed for model reconstruction)
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128),
    batch_tfms=aug_transforms() + [Normalize.from_stats(*imagenet_stats)]
).dataloaders(path, bs=64)

# 🏗 Rebuild the model architecture (must match training)
learn = vision_learner(dls, resnet34, metrics=accuracy)

# 🔄 Load the trained weights from .pth file
learn.load('/content/models/animal_classifier_resnet34')  # Do NOT include .pth extension

# 🖼️ Load an image for prediction
img_path = 'deer.jpeg'  # Replace with your image path
img = PILImage.create(img_path)

# 🔍 Predict the class
pred_class, pred_idx, probs = learn.predict(img)

# 🖼️ Display results
print(f"Predicted Class: {pred_class}")
print(f"Confidence: {probs[pred_idx]:.4f}")

# 📸 Show the image with prediction
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {pred_class} ({probs[pred_idx]:.2%})', fontsize=14)
plt.show()
