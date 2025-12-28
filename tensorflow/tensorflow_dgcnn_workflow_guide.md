# DGCNN 点云分类 TensorFlow 版完整流程指南

## 1. 环境准备

建议使用 Python 3.6~3.7（TensorFlow 1.15 最佳兼容），并使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install "tensorflow==1.15" numpy scipy h5py scikit-learn tqdm
```

> 注意：TensorFlow 1.x 不支持 Python 3.8 及以上版本，推荐用 3.6/3.7。

## 2. 数据集准备

1. 手动下载 `modelnet40_ply_hdf5_2048.zip` 数据集。
2. 解压后，将 `modelnet40_ply_hdf5_2048` 文件夹放到：

```
dgcnn-master/tensorflow/data/modelnet40_ply_hdf5_2048
```

## 3. 训练模型

进入 tensorflow 目录，运行训练脚本：

```bash
cd dgcnn-master/tensorflow
python train.py
```

- 日志和模型会保存在 `train_results/` 目录下。
- 如需快速测试流程，可尝试修改 `train.py` 里的 epoch、batch_size 等参数。

## 4. 评估模型

训练完成后，运行评估脚本：

```bash
python evaluate.py
```

- 评估结果会输出到终端。
- 可根据 `evaluate.py` 里的参数指定模型路径、测试集等。

## 5. 分析评估结果

- 查看终端输出，关注 test accuracy、loss 等指标。
- 如需更高准确率，可调整参数（如 batch_size、epoch、learning rate 等）重新训练。
- 可用 scikit-learn、matplotlib 等工具进一步分析、可视化结果。

## 6. 进阶操作

- 用自己的点云数据做推理：可参考 `evaluate.py` 加载模型部分，写自定义推理脚本。
- 尝试不同参数、不同数据集。
- 结果可用于论文复现、课程作业、学术研究等。

---

如遇到任何报错或有新需求，欢迎随时提问！
