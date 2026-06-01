**🐾 Animal Image Classifier - ResNet18**<br><br>
**📌 Project Overview**<br>
This project is a deep learning-based "animal image classifier" trained using the **Animals-90 dataset**. <br>
The model utilizes "Transfer Learning" with a "ResNet34" backbone to accurately classify images into one of "90 different animal categories". <br>
<br>
**🛠 Model Details**<br>
**🧠 Architecture:** ResNet34 (Pretrained on ImageNet) <br>
**📚 Framework:** FastAI (PyTorch-based) <br><br>
**📝 Training Approach:** <br>
✅ Used **Transfer Learning** to leverage pre-trained weights <br>
✅ **One-Cycle Learning Rate Policy** for better optimization <br>
✅ **Mixed Precision Training** (to_fp16()) for faster performance <br>
✅ **Progressive Resizing** (Start with small images, increase size gradually) <br>
✅ **Custom Data Augmentations** (Rotation, Zoom, Lighting Adjustments) <br>
✅ **Early Stopping & Best Model Saving** <br><br>
**🎯 Validation Accuracy:** High accuracy achieved with optimized training settings <br>
**⏳ Training Time:** Reduced using "mixed precision training" <br>
**🔍 Misclassification Analysis:** Used "FastAI’s interpretation tools" to visualize top losses <br><br>
**📂 Dataset Details**<br>
**🗂 Dataset Name:** Animals-90 <br>
**📌 Source:** Kaggle: iamsouravbanerjee/animal-image-dataset-90-different-animals <br>
**📑 Total Categories:** 90 Animal Classes <br>
**📊 Split Ratio:** 80% Training, 20% Validation <br><br>
**🔍 Key Features**
✔ **Highly accurate model** with improved training strategies <br>
✔ **Supports new image predictions** with ease <br>
✔ **Optimized for speed & performance** using mixed precision <br>
✔ **Reduces overfitting** with MixUp & progressive resizing <br><br>

**🚀 Usage**
📌 **Load the model** and classify images instantly <br>
📌 **Supports deployment** in web apps or mobile applications <br>
📌 **Can be extended** to classify more animal categories <br>

**Accuracy 82.96%**
