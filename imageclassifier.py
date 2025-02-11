import kagglehub
from fastai.vision.all import *

# Download dataset
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
path = Path('/root/.cache/kagglehub/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/versions/5/animals')

# Define custom augmentations
custom_transforms = aug_transforms(
    max_rotate=15,       # Reduce rotation to avoid over-distortion
    max_zoom=1.3,        # Lower zoom for better context retention
    p_affine=0.7,        # Affine transform probability
    p_lighting=0.7,      # Slightly increase lighting changes
)

# Use Progressive Resizing: Start small, then increase resolution
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128),  # Start small
    batch_tfms=[custom_transforms, Normalize.from_stats(*imagenet_stats)]
).dataloaders(path, bs=64)  # Increase batch size for efficiency

# Preview data
dls.show_batch(max_n=15)

# Load model with a deeper architecture
learn = vision_learner(dls, resnet34, metrics=accuracy, cbs=MixUp(0.2)).to_fp16()  # Mixed Precision

# Find best learning rate
learn.lr_find()

# Train with one-cycle policy & early stopping
learn.fine_tune(
    10,   # Reduce epochs for faster training
    base_lr=3e-3, 
    freeze_epochs=2,  # Freeze early layers initially
    cbs=[
        EarlyStoppingCallback(monitor='valid_loss', patience=3), 
        SaveModelCallback(monitor='valid_loss', fname='animal_classifier_resnet34')
    ]
)

# Load best model
learn.load('animal_classifier_resnet34')

# Interpret results
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(10,10))

# Export model
learn.export('animal_classifier_resnet34.pkl')
