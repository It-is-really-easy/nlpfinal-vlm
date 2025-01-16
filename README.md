
# VLM (Vision-Language Model) for Change Detection in Remote Sensing Images

## 作者: 李泽鸣

### 简介

本项目实现了一个基于视觉语言模型（VLM）的遥感图像变化检测系统。模型结合了CLIP-ViT-Large-Patch14进行图像特征提取，并通过Qwen-7B-Chat模型生成变化检测结果。我们设计了两种Encoder架构：一种是基于MLP（多层感知机），另一种是引入自注意力机制（Self-Attention），用于处理图像的特征融合。最终，模型能够检测遥感图像中的变化，并生成相关的文本描述。

### 环境配置

以下是配置环境的步骤：

1. **创建conda环境**：
   使用`conda`创建一个Python 3.10的虚拟环境：

   ```bash
   conda create -n vlm python=3.10


2. **安装依赖**： 激活环境并通过`pip`安装项目所需的依赖库：

   ```bash
   conda activate vlm
   pip install -r requirements.txt
   ```

3. **数据集准备**： 解压数据集到`dataset`目录。确保数据集格式与项目要求一致。

   ```bash
   mkdir dataset
   # 将数据集文件解压到 dataset 目录中
   ```

4. **训练模型**： 运行`train.sh`脚本开始训练。此脚本会自动处理训练过程，包含模型训练和必要的调优步骤。

   ```bash
   bash train.sh
   ```

5. **测试模型**： 训练完成后，可以通过以下两种方式测试模型：

   - `test_single`：测试单张图像，生成对应的变化检测结果。
   - `test_all`：进行整体测试，评估模型在多张图像上的表现和各项指标。

   运行测试脚本：

   ```bash
   bash test_single.sh
   bash test_all.sh
   ```

### 参考资料

本项目参考了 [VLM-learning](https://github.com/WatchTower-Liu/VLM-learning) 中的部分方法与技术。该项目为视觉语言模型的学习提供了大量的启发，特别是在图像和文本特征融合的处理上。

### 运行注意事项

- 确保环境中已安装`PyTorch`以及其他必要的深度学习库。
- 数据集的路径和格式需要与项目一致，确保能够被正确加载。
- 在训练和测试过程中，可以根据硬件配置调整批次大小（batch size）等参数。
