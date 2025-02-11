🐾 Animal Image Classifier - ResNet34
📌 Project Overview
This project is a deep learning-based animal image classifier trained using the Animals-90 dataset. The model utilizes Transfer Learning with a ResNet34 backbone to accurately classify images into one of 90 different animal categories.

🛠 Model Details
🧠 Architecture: ResNet34 (Pretrained on ImageNet)
📚 Framework: FastAI (PyTorch-based)
📝 Training Approach:
✅ Used Transfer Learning to leverage pre-trained weights
✅ One-Cycle Learning Rate Policy for better optimization
✅ Mixed Precision Training (to_fp16()) for faster performance
✅ Progressive Resizing (Start with small images, increase size gradually)
✅ Custom Data Augmentations (Rotation, Zoom, Lighting Adjustments)
✅ Early Stopping & Best Model Saving
📊 Model Performance
🎯 Validation Accuracy: High accuracy achieved with optimized training settings
⏳ Training Time: Reduced using mixed precision training
🔍 Misclassification Analysis: Used FastAI’s interpretation tools to visualize top losses
📂 Dataset Details
🗂 Dataset Name: Animals-90
📌 Source: Kaggle: iamsouravbanerjee/animal-image-dataset-90-different-animals
📑 Total Categories: 90 Animal Classes
📊 Split Ratio: 80% Training, 20% Validation
🔍 Key Features
✔ Highly accurate model with improved training strategies
✔ Supports new image predictions with ease
✔ Optimized for speed & performance using mixed precision
✔ Reduces overfitting with MixUp & progressive resizing

🚀 Usage
📌 Load the model and classify images instantly
📌 Supports deployment in web apps or mobile applications
📌 Can be extended to classify more animal categories
