# STT-RCA

面向**多变量时间序列、已确认的单根异常完整事件**，仅由 `x` 与 `valid` 对根通道排序。
本实现依据 `implementation-001/refine-logs/FINAL_PROPOSAL.md` 的 STT-RCA 规格，**从头初始化**。
模型是共享轴向 Transformer，不加载 TimeRCD 编码器或权重。独立的数据生成仓库只提供冻结正常世界、数值重放和数据访问。

本轮交付是代码与正确性验证，**不启动、排队或自动恢复训练**。没有已训练模型或泛化收益结论。

## 架构

- 逐事件、逐通道在有效 float32 观测上升为 float64，计算 linear median/IQR；floor 为 `1e-4`，不裁幅。无效值在线性层前清零。
- patch=16；拼接16个值和16个有效位。共享 patch 投影及固定 sin/cos 时间位置，无通道身份或拓扑位置。
- 8层轴向 stage：时间 MHA、通道 MHA、FFN，pre-LN，宽512，8头，FFN2048，dropout=0，双向注意力。时间轴先置换再 reshape；全空 attention 行不进入 softmax。
- H 是 `[B,P,C,256]` 上下文网格，U 是 `[B,P,C,128]` 源网格。按非空 patch 等权池化 U，共享 `LN→Linear128/64→GELU→Linear64/1` 根头。
- 训练解码器4层，宽256，4头，FFN1024；源投影无 bias；每 token 输出16点。两次完整 FP32 前向后相减，Y0 被 base/trans 共享且不 detach。
- 混合精度只由调用方控制主干；H/U、根头、解码器和损失显式 FP32。默认评价全 FP32，关闭 TF32。
- 训练总参数 **38,106,897**；导出根排序器 **33,724,417**，移除 H projection 和 decoder。

公共推理 API 为 `ranker.predict(x, valid)`；原始候选通道完全无观测时抛出 `UNSUPPORTED_UNOBSERVED_CHANNEL`。
batch 的额外 padding 通过先逐事件归一化、再拼接 `NormalizedObservation` 表示，不能把原始全缺失候选冒充 padding。

## 目标与接口

`STTObjective` 封装完整 A/B/F 图，DDP 应包装该模块，而不是在 DDP 外调用各个子模块。
根 CE 权重为 A/B/F = 0.25/0.25/0.5；F 使用自己的观测统计。正常及响应目标使用 B 的观测坐标。
移植项对每个 pair 先计算有效标量 MSE / (目标平均平方能量 + epsilon)，再按 pair 平均。
`TrainingPair` 中的标签和特权目标只供训练目标函数使用；普通 encoder/decoder 没有这些输入接口。

已提供 STT、Direct、Delete-trans、Oracle、payload supervision、普通 masked reconstruction 的目标实现。
Kappa 与 cross/within 的专用匹配数据研究不通过改名字冒充普通配对；它们仍需下一阶段冻结专用 manifest 并单独实施。

`training.py` 提供 AdamW 参数分组、warmup/cosine、精确96条全局分配、`no_sync` 累积和各 rank RNG/游标/契约一致的检查点。
这些是显式调用的训练组件，不包含后台作业或自动实验编排。真正学习效果、GPU/NCCL和吞吐验收仍待后续训练阶段。

## 配置与现有数据

`configs/specification/` 保存原始模型/先导/评价配置；`specification_inputs.json` 记录四份输入的 SHA-256。
`configs/stt_rca_six_groups.yaml` 是本轮确认的六组版本：C=8/12/16/20/32/48。评价使用已冻结的120个验证系统、240个测试系统，两条轨共46,080个 own-event。
模型天然支持任意真实 T；配对依据现有语料的真实长度支持，不将原 YAML 的两点长度示例重新采样覆盖生成后的分布。

主数据包括30亿训练观测标量、40.06016亿额外评价标量和192万主 pair。先导需另建独立192,000条 manifest，不能把主 manifest 的前192,000行叫独立先导。
`prepare.py` 提供独立 Q_adm 抽样流、固定排列及与64MiB缓存一致的块计划；不按响应或难度筛样本。
`FrozenPairDataset` 只将观测交给网络，核对根标签、双亲谱系、cell、坐标及缓存边界；可显式指定独立CPU缓存 producer。
CPU producer 不导入 torch，读取当前进程映像的 `VmHWM`，不把 Linux 继承的父进程 `ru_maxrss` 错当自身内存。
缓存继续使用生成器的16GiB节点上限、pin/LRU/原子提交，绝不缓存 learned U/H。数据集谱系固定到依赖中的生成提交。

CPU producer 每次冷启动还会校验既有块，当前实现以完整性优先，缓存吞吐尚未实测；不得据此承诺训练速度。

## 使用与验证

安装模型（数据适配可选）：

```bash
python -m pip install -e .
# 需要读取生成语料时：python -m pip install -e '.[data]'
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
stt-rca inspect --config configs/stt_rca_six_groups.yaml
```

没有训练命令会由安装、导入、inspect 或测试自动执行。测试只做前向、梯度与恢复契约核验，不执行优化器更新。
部署使用 `export_ranker(objective, path)` / `load_ranker(path)`，或 CLI `predict` 读取仅含 x(float32)、valid(bool) 的 NPZ。
随机初始化导出只用于一致性测试，不能作为训练完成的模型发布。

`metrics.py` 使用实际 float32 logits 的精确并列期望 MRR/Top-1。
主表先 system 内平均，再 system 间平均，两轨等权选择；system cluster bootstrap 按 C 分层，两轨/模型共用系统抽样。
`evaluation.py` 保留全部合法 own-event，另报 mode、C/T、sink 与物理/量化/标准化可见性切片。

原理接口依据提供的 STT-RCA 文档；activation checkpoint 使用 PyTorch 的非重入接口，见[官方文档](https://docs.pytorch.org/docs/2.5/checkpoint.html)。
许可证沿用本仓库已有 MIT。数据生成仓库及其上游许可证独立保留。
