# YOLO26 IoU 损失函数扩展指南

> 基于 Ultralytics YOLO26 代码库。实验环境：12 轮短训练，小目标数据集（物体约占画幅 0.05–0.10）。
> 结论：所有扩展的 IoU 变体均未超过原版 CIoU，WIoU v3 最接近（差约 0.06 mAP）。

---

## 文件索引

| 文件 | 作用 |
|------|------|
| `ultralytics/utils/loss.py` | `BboxLoss` 类 —— 训练时的 box 损失计算（**WIoU v3、Inner-IoU、Alpha-IoU**） |
| `ultralytics/utils/metrics.py` | `bbox_iou()` 函数 —— 单次 IoU 评分计算（**MPDIoU、EIoU、WIoU v1**） |
| `ultralytics/cfg/default.yaml` | 损失相关超参数配置 |

---

## 损失函数一览

### 0. 原版 CIoU（基线）

**论文**：Zheng et al., Distance-IoU Loss (AAAI 2020)

**公式**：

$$\text{CIoU} = IoU - \frac{\rho^2(b, b_{gt})}{c^2} - \alpha v$$

$$\alpha = \frac{v}{1 - IoU + v}, \quad v = \frac{4}{\pi^2}\left(\arctan\frac{w_{gt}}{h_{gt}} - \arctan\frac{w}{h}\right)^2$$

```
三部分：IoU + 中心距离惩罚 + 宽高比一致性惩罚
```

**代码位置**：[`metrics.py:171-175`](ultralytics/utils/metrics.py#L171-L175)

---

### 1. MPDIoU —— 角点距离惩罚

**论文**：Ma & Xu, arXiv 2307.07662

**公式**：

$$\text{MPDIoU} = IoU - \frac{d_1^2 + d_2^2}{c^2}$$

- $d_1^2$：预测框与 GT 框**左上角**距离平方
- $d_2^2$：预测框与 GT 框**右下角**距离平方
- $c^2$：最小外接矩形（凸包）对角线平方

**原理**：用角点距离替代中心距离 + 宽高比，计算更简单。角点对齐直接约束了框的形状和位置。

**代码位置**：[`metrics.py:179-189`](ultralytics/utils/metrics.py#L179-L189)

**如何在 loss.py 中启用**：将 [`BboxLoss.forward()`](ultralytics/utils/loss.py#L216) 中 `CIoU=True` 改为 `MPDIoU=True`。

**实验结果**：❌ 比 CIoU 差。原因：角点在小目标上标注噪声大，距离平方对小框天然不友好。

---

### 2. EIoU —— 解耦宽高惩罚

**论文**：Zhang et al., Focal and Efficient IOU Loss, arXiv 2101.08158

**公式**：

$$\text{EIoU} = IoU - \frac{\rho^2(b, b_{gt})}{c^2} - \frac{(w - w_{gt})^2}{c_w^2} - \frac{(h - h_{gt})^2}{c_h^2}$$

```
三部分：IoU + 中心距离惩罚 + 独立宽度惩罚 + 独立高度惩罚
```

**与 CIoU 的区别**：CIoU 用 $\arctan(w/h)$ 比值惩罚宽高比，对小目标不稳定（±1px 就能让比值剧烈变化）。EIoU 改为宽度和高度各自做 L2 惩罚，梯度更平稳。

**代码位置**：[`metrics.py:190-202`](ultralytics/utils/metrics.py#L190-L202)

**如何在 loss.py 中启用**：将 `CIoU=True` 改为 `EIoU=True`。

**实验结果**：❌ 不如 CIoU。可能原因为网格坐标系下宽高惩罚的归一化方式与 CIoU 的 $\alpha v$ 自适应相比缺乏动态调节。

---

### 3. WIoU v1 —— 乘法中心距离注意力

**论文**：Tong et al., Wise-IoU, arXiv 2301.10051

**公式**：

$$\text{WIoU v1} = IoU \cdot \exp\left(-\frac{\rho^2(b, b_{gt})}{c^2}\right)$$

```
乘法模型：IoU × 中心距离注意力门控
```

**与其他 IoU 的本质区别**：

| 类型 | 公式模式 | 梯度行为 |
|------|---------|---------|
| DIoU/CIoU/EIoU/MPDIoU | **加法惩罚**：$IoU - p$ | 惩罚梯度恒定 |
| **WIoU v1** | **乘法注意力**：$IoU \cdot e^{-d}$ | 中心越远梯度越强 |

**原理**：`exp(-ρ²/c²)` 作为"注意力门控"——中心距离大时门控接近 0，loss ≈ IoU，梯度来自 IoU 本身；中心距离小时门控接近 1，loss ≈ IoU，此时已对齐不再需要强梯度。天然实现"先粗后细"。

**代码位置**：[`metrics.py:203-216`](ultralytics/utils/metrics.py#L203-L216)

**实验结果**：❌ 早期 4-6 轮高于 CIoU，后期被反超。`exp` 门控在中心对齐后接近 1.0，loss 退化近似纯 IoU，失去宽高精修能力。

---

### 4. WIoU v3 —— 离群度自适应注意力 ⭐

**论文**：同上（Wise-IoU 进阶版）

**公式**：

$$\text{Loss} = r \cdot \left[1 - IoU \cdot \exp\left(-\frac{\rho^2}{c^2}\right)\right]$$

$$r = \frac{\beta}{\delta \cdot \alpha^{\beta - \delta}}, \quad \beta = \frac{1 - IoU}{\overline{L_{IoU}}}$$

| 符号 | 含义 |
|------|------|
| $\beta$ | 离群度：当前锚框的 IoU loss 与滑动平均的比值。β>1 表示该锚框"比平均难" |
| $\overline{L_{IoU}}$ | 滑动平均 IoU loss（动量 0.99） |
| $r$ | v3 注意力因子：中等离群度获得最大 r，极好/极差锚框 r 衰减 |
| $\alpha, \delta$ | 控制 r 曲线的形状（默认 α=1.9, δ=1.0） |

**与 v1 的关键区别**：v1 后期梯度消失是因为中心对齐后 exp 门控接近 1.0。v3 追加的 r 因子在后期反而不衰减——此时 IoU 高 → β 低 → r 高 → 持续驱动宽高精修。

```
r 因子的行为（α=1.9, δ=1.0）：
  β=0.5 (高质量) → r≈0.6  （抑制，防止过拟合）
  β=1.0 (中等)   → r≈1.0  （正常）
  β=2.0 (困难)   → r≈0.7  （抑制，防止极端离群值主导）
  β=0.3 (极高质量) → r≈0.4 （后期 IoU 高时本应更大，但 δ=1.0 限制）
```

**代码位置**：[`loss.py:189-214`](ultralytics/utils/loss.py#L189-L214)（BboxLoss 内）

**启用方式**：
```bash
yolo train model=yolo26n.yaml data=your_data.yaml wiou=True
```

**可调参数**（在 `default.yaml` 或命令行）：
```yaml
wiou: True
wiou_momentum: 0.99   # 滑动平均动量（越小适应越快，12 轮建议 0.9）
wiou_alpha: 1.9       # r 曲线尖锐度（越大峰值越尖）
wiou_delta: 1.0       # r 曲线拐点（β=δ 时 r=1）
```

**实验结果**：⭐⭐ 最接近 CIoU（差约 0.06 mAP）。问题在于 12 轮训练不足以让滑动平均稳定，r 因子全程在偏移。

---

### 5. Inner-IoU —— 中心缩放框 ⭐

**论文**：Zhang et al., Inner-IoU, arXiv 2311.02877

**公式**：

```
对于 ratio r（如 0.7）：
  cx = (x1 + x2) / 2,  cy = (y1 + y2) / 2
  新宽度 = 原宽度 × r,  新高度 = 原高度 × r
  内框 = [cx - new_w/2, cy - new_h/2, cx + new_w/2, cy + new_h/2]

用内框代替原框计算 IoU（可叠加任意基础 IoU）
```

```
r=0.7 → 32×32 框 → 22.4×22.4 内框
边缘 4.8px（含标注噪声）被裁掉
```

**原理**：GT 标注在小目标上 ±1px 的偏差就是 3-5% 的误差。内框裁剪掉边界噪声，聚焦在中心可靠区域计算相似度。无滑动平均、无自适应机制——纯几何变换，从头稳到尾。

**代码位置**：[`loss.py:172-187`](ultralytics/utils/loss.py#L172-L187)（BboxLoss 内）

**启用方式**：
```bash
yolo train model=yolo26n.yaml data=your_data.yaml inner_ratio=0.7
```
叠加到 WIoU 上：
```bash
yolo train model=yolo26n.yaml data=your_data.yaml wiou=True inner_ratio=0.7
```

**可调参数**：
```yaml
inner_ratio: 0.7   # 0.7–0.8 适合小目标，0.0 = 关闭
```

**实验结果**：❌ 不如 CIoU。可能对 0.05-0.10 级目标来说内框太小（r=0.7 时 16×16 变为 11×11），损失信息过多。

---

### 6. Alpha-IoU —— 幂变换梯度再分配

**论文**：He et al., Alpha-IoU, arXiv 2110.13675

**公式**：

$$L_{\alpha} = (1 - CIoU)^{\alpha}$$

```
α=1: 原版 CIoU loss
α=3: 高 IoU 匹配的 loss 被放大，低 IoU 匹配被压缩
```

**梯度放大因子**：$\nabla L_\alpha \propto \alpha \cdot (1 - CIoU)^{\alpha-1}$

| 锚框 IoU | α=1 梯度 | α=3 梯度 | α=3 / α=1 |
|----------|---------|---------|-----------|
| 0.3 (差) | 0.70 | 0.19 | **0.27×** ← 压制 |
| 0.7 (中) | 0.30 | 0.44 | **1.47×** ← 放大 |
| 0.9 (好) | 0.10 | 0.24 | **2.43×** ← 强力 |

**原理**：早期低 IoU 锚框多，不需要被放大；后期高 IoU 锚框多但梯度小，需要被放大。幂变换天然实现了这个"先粗后细"的梯度分配——无额外状态维护，纯静态变换。

**代码位置**：[`loss.py:217-222`](ultralytics/utils/loss.py#L217-L222)（BboxLoss 内，仅 3 行）

**启用方式**：
```bash
yolo train model=yolo26n.yaml data=your_data.yaml alpha_iou=3.0
```
叠加到 Inner-IoU 上：
```bash
yolo train model=yolo26n.yaml data=your_data.yaml alpha_iou=3.0 inner_ratio=0.7
```

**可调参数**：
```yaml
alpha_iou: 3.0   # ≥1.0，1.0=关闭。论文常用 3.0
```

**实验结果**：❌ 不如 CIoU。12 轮训练时间太短，alpha_iou 压低早期梯度的代价超过后期精修的收益。

---

## 实验结果汇总

| IoU 类型 | mAP vs CIoU | 早期表现 | 后期表现 | 备注 |
|----------|------------|---------|---------|------|
| **CIoU（原版）** | 基准 | — | — | 当前最优 |
| MPDIoU | - >0.01 | 差 | 差 | 角点噪声对小目标影响大 |
| EIoU | - >0.01 | 差 | 差 | 网格坐标下宽高惩罚效果不佳 |
| WIoU v1 | - >0.01 | **好** | 被反超 | exp 门控后期退化 |
| **WIoU v3** ⭐ | **-0.06** | 一般 | 最接近 | 滑动平均在短训练中不稳定 |
| Inner-IoU | - >0.01 | 差 | 差 | 裁剪掉的信息 > 噪声收益 |
| Alpha-IoU | - >0.01 | 差 | 一般 | α=3 压低早期梯度，12 轮不够恢复 |

---

## 可能的方向（未尝试）

| 方向 | 思路 | 适合场景 |
|------|------|---------|
| **组合 Inner-IoU + Alpha-IoU** | r=0.85（轻裁剪）+ α=2（温和放大），两条路径同时生效 | 标注噪声 + 短训练 |
| **SIoU** | 在距离惩罚前加角度代价项，收敛更快 | 短训练友好 |
| **Focaler-IoU** | 阈值聚焦中等质量匹配，忽略极好/极差样本 | 噪声标注 |
| **Focal-EIoU** | EIoU + Focal Loss 的 gamma 调制 | 困难样本挖掘 |
| **NWD** | 框建模为 2D 高斯分布，Wasserstein 距离 | 物体 < 0.05（极小小目标） |
| **增加训练轮数** | WIoU v3 需要 > 50 轮让滑动平均稳定 | WIoU v3 潜力最大 |

---

## 使用示例

```bash
# 原版 CIoU（默认）
yolo train model=yolo26n.yaml data=custom.yaml epochs=12

# WIoU v3
yolo train model=yolo26n.yaml data=custom.yaml epochs=12 wiou=True

# Inner-IoU (CIoU 基底)
yolo train model=yolo26n.yaml data=custom.yaml epochs=12 inner_ratio=0.7

# Alpha-IoU (CIoU 基底)
yolo train model=yolo26n.yaml data=custom.yaml epochs=12 alpha_iou=3.0

# 组合：Inner-IoU + Alpha-IoU（两条路径同时生效）
yolo train model=yolo26n.yaml data=custom.yaml epochs=12 inner_ratio=0.7 alpha_iou=3.0

# 组合：Inner-IoU + WIoU v3
yolo train model=yolo26n.yaml data=custom.yaml epochs=12 wiou=True inner_ratio=0.7
```

---

## 架构说明

```
训练调用链：
  trainer.py: model(batch)
    → tasks.py: DetectionModel.loss()
      → loss.py: E2ELoss.__call__()
        ├── one2many: v8DetectionLoss.loss()
        └── one2one:  v8DetectionLoss.loss()
              → get_assigned_targets_and_loss()
                → self.bbox_loss(pred_dist, pred_bboxes, ...)  ← BboxLoss.forward
                      ├── Inner-IoU 框缩放  (if inner_ratio > 0)
                      ├── WIoU v3 路径       (if self.wiou)
                      └── CIoU + Alpha-IoU   (else, 默认)
                → bbox_iou(pred, target, CIoU/MPDIoU/EIoU/WIoU ...)
                      ↑
                      metrics.py 中的 IoU 评分函数
```

关键点：
- **`BboxLoss`**（`loss.py`）控制损失计算和梯度放大 → WIoU v3、Inner-IoU、Alpha-IoU 在此
- **`bbox_iou()`**（`metrics.py`）仅计算 IoU 评分 → MPDIoU、EIoU、WIoU v1 在此
- Inner-IoU 和 Alpha-IoU 可以**叠加**到任意基础 IoU 上
- WIoU v3 是独立路径，不与其他 IoU 叠加（内部已调用 `bbox_iou` 算基础 IoU）
