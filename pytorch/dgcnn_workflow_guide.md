# DGCNN 点云分类完整流程指南

## 1. 环境准备

```bash
# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install torch numpy h5py scikit-learn
```

## 2. 数据集准备

1. 手动下载 `modelnet40_ply_hdf5_2048.zip` 数据集。
2. 解压后，将 `modelnet40_ply_hdf5_2048` 文件夹放到：

```
dgcnn-master/pytorch/data/modelnet40_ply_hdf5_2048
```

## 3. 流程测试（可选）

先用极小参数测试代码能否跑通：

```bash
cd dgcnn-master/pytorch
python main.py --exp_name=test_run --model=dgcnn --num_points=128 --k=5 --use_sgd=True --batch_size=2 --test_batch_size=2 --epochs=1
```

## 4. 正式训练

推荐参数（以 1024 点为例）：

```bash
python main.py --exp_name=dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --batch_size=32 --epochs=250
```

- 日志和模型会保存在 `checkpoints/dgcnn_1024/` 目录下。
- 如果用 CPU，建议先用较小 batch/epoch 测试。

## 5. 评估模型

训练完成后，评估模型性能：

```bash
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=checkpoints/test_run/models/model.t7
```

```test
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=128 --k=5 --use_sgd=True --eval=True --model_path=checkpoints/test_run/models/model.t7
```

- 评估结果会输出到终端和 `checkpoints/dgcnn_1024_eval/run.log`。

## 6. 分析评估结果

- 查看终端输出和 `run.log`，关注 test acc、test avg acc 等指标。
- 如需更高准确率，可调整参数（如 batch_size、k、emb_dims、epochs 等）重新训练。
- 可用 scikit-learn、matplotlib 等工具进一步分析、可视化结果。

## 7. 进阶操作

- 用自己的点云数据做推理：可参考 `main.py` 加载模型部分，写自定义推理脚本。
- 尝试不同模型（如 PointNet）、不同参数、不同数据集。
- 结果可用于论文复现、课程作业、学术研究等。

---

如遇到任何报错或有新需求，欢迎随时提问！
