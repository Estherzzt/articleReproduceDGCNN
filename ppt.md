推荐 PPT 结构（10–14 页）
我给你一个实用结构，每一页都写上「要讲的重点」。
⭐ 封面页
Title：DGCNN Paper Reproduction — Understanding & Implementation
Name / Date
Mentor name（可选）
👉 显得正式
⭐ Page 1 — Background & Task Objective
内容要简短：
What is the task?
3D Point Cloud Classification / Segmentation
Why DGCNN matters?
Local geometric relationships
Dynamic neighborhood learning
My task:
Reproduce code
Understand model
Run experiments & report insights
👉 用 3～4 bullet，不要堆字
⭐ Page 2 — Paper Overview（核心思想）
建议画 一张逻辑图（你自己画即可）：
包括：
输入：点云 
N
×
3
N×3
KNN 构图
EdgeConv 提取边特征
多层堆叠
Global pooling
并加一句总结：
Key idea: Learn features on dynamic graphs instead of static neighborhoods.
👉 这一页是“展示你理解论文”的关键
⭐ Page 3 — EdgeConv Explained（必须讲清）
内容：
edge feature = 
h
(
x
i
,
x
j
−
x
i
)
h(x 
i
​	
 ,x 
j
​	
 −x 
i
​	
 )
Preserve:
local geometry
translation awareness
Aggregation via max-pooling
右侧放一小张示意框图即可。
👉 这一页是 mentor 最可能问问题的地方
⭐ Page 4 — Dynamic Graph vs Static Graph
两栏对比表：
Static Graph	Dynamic Graph
fixed based on xyz	rebuilt per layer
same neighbors	feature-space neighbors
weak semantics	stronger semantics
最后一句总结：
Dynamic graph helps high-level semantic grouping.
👉 体现“理解到位，而不是机械复现”
⭐ Page 5 — Repository Structure (PyTorch Version)
列出 pytorch 目录对应关系：
Component	File
Data loading	data.py
Model (EdgeConv)	model.py
Training loop	main.py
Utilities	util.py
再加一句：
I focused on PyTorch for main reproduction and understanding.
👉 这是“代码层理解”的起点
⭐ Page 6 — Code–Paper Mapping（加分页🔥）
做一个非常加分的表：
Paper Concept	Code Implementation
EdgeConv	get_graph_feature() + EdgeConv()
Dynamic KNN	knn(x, k) called in each layer
Feature aggregation	concatenation across EdgeConv layers
Global representation	x.max(dim=2)
👉 这一页导师非常喜欢
⭐ Page 7 — Engineering Work I Did（非常关键）
这一页一定要写你“自己做了什么”：
Manual dataset preparation
Modify shell commands → cross-platform (mac/Windows)
CPU-only environment setup
Small-scale sanity test
Logging & reproducibility scripts
👉 这是把你从“学生”升级到“工程实习生”的地方
⭐ Page 8 — PyTorch Experiment Setup（训练中可留空）
内容可以先写框架，结果明天补：
ModelNet40
num_points=1024
k=20
optimizer: SGD
batch size / epochs
留一个占位：
Training in progress…
Test Accuracy:  xx.xx %  (to be filled)
Loss curve: (graph to insert)
👉 你可以等训练结束再补图
⭐ Page 9 — TensorFlow Version (Brief Study)
写三点即可：
Static graph (session-based) pipeline
Similar model structure, different framework style
I ran a small-scale experiment to compare behavior
可以写：
I mainly used TF version for structural comparison, not full reproduction.
👉 展现 breadth，不用花太多篇幅
⭐ Page 10 — Comparison PyTorch vs TensorFlow（简短）
PyTorch	TensorFlow
easier debugging	harder debugging
dynamic graph	static graph
better for research iteration	closer to early original code
I used for reproduction	I used for reference
👉 导师会觉得你“有判断、有选择”
⭐ Page 11 — What I Learned
写反思型总结：
Understanding graph-based feature learning
Difference between implementation & paper abstraction
Practical issues in reproducibility
Trade-off: CPU-only vs speed
👉 是一页 成熟、诚实、有思考 的总结
⭐ Page 12 — Next Steps（导师讨论入口）
写 3～4 条即可：
Ablation:
static vs dynamic graph
Try segmentation experiment
Visualize neighborhood evolution
Try larger-scale full training
👉 给 mentor 提供“讨论话题”
🛠️ 你现在可以这样安排工作
今晚 / 明早：
1️⃣ 开始跑 PyTorch 1024 full training（CPU版）
2️⃣ 跑一个 小规模 TensorFlow run
3️⃣ 同时完成 PPT 的前 10 页（不含结果）
等训练完成：
4️⃣ 把：
test accuracy
loss curve
few logs / screenshots
填入 Page 8
