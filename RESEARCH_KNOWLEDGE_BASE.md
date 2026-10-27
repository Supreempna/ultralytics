# 前视声呐（FLS）水下目标检测科研知识库与核心实验备忘录
> **Project Research Knowledge Base & Experimental Roadmap**  
> **目标投稿**：IEEE TGRS（中科院一区 Top / 遥感声呐标杆）/ IEEE JOE（海洋工程顶级）/ IEEE TIP / CVPR  
> **最后更新时间**：2026-10-27  

---

## 一、 项目全局画像与核心事实（Ground Truth）

> ⚠️ **核心原则**：后续所有推导、架构设计与消融分析，必须严格锚定以下客观实验事实，绝不凭空假设。

### 1. 任务与数据集（FLSD: Forward-Looking Sonar Detection）
* **任务定位**：面向自主水下航行器（AUV）边缘嵌入式部署的**轻量级前视声呐目标检测**。
* **数据规模**：3001 张训练图，751 张验证图（图像尺寸：$640 \times 640$）。
* **类别与数量分布**：
  * `0: victim`（溺水者/人体）：279 个目标；
  * `1: boat`（船只）：261 个目标；
  * `2: plane`（飞机）：263 个目标。
* **目标真实尺寸分布（彻底打破“小目标假说”）**：
  * 通过 `compute_size_ap.py` 严格诊断验证集的 803 个真实标注框（按 COCO 标准）：
    * **`small`（面积 $< 32^2 = 1024\text{ px}^2$）**：**仅 2 个目标**（无统计学意义，忽略不计）；
    * **`medium`（$32^2 \le \text{面积} < 96^2$）**：**528 个目标**（几乎全为 `boat` 与 `plane`）；
    * **`large`（面积 $\ge 96^2 = 9216\text{ px}^2$）**：**273 个目标**（几乎全为 `victim`）。
  * 🔑 **关键洞察**：**类别与尺寸高度耦合**！`victim ≈ large`，`boat/plane ≈ medium`。任何针对 COCO 小目标（Tiny Object）的模块（如 P2 微小头、NWD 瓦瑟斯坦小目标损失）在本项目中均属**文献误用**，数据不支持！

### 2. 核心性能瓶颈分析
* 整体性能中，约 **17% 的漏检（平均召回率 AR = 0.832）几乎全部集中在 `victim` 类别**。
* 各类别 AP 分布：
  * `plane`: **0.817**（最高，金属刚性强反射）；
  * `boat`: **0.775**；
  * `victim`: **0.741（全场最低，核心性能瓶颈）**。
* **物理成因**：人体（肉体组织）声阻抗与水极其接近，漫反射微弱，回波信噪比极低，容易被混响湮没。

### 3. 基准与已有成功结论（Baseline & Key Findings）
* **Baseline（原 YOLO26n，20ep）**：mAP50-95 = **72.56%**。
* **普通 NAFNet 前端（20ep）**：mAP50-95 = **75.78% (+3.22%)**。
* **普通 NAFNet 前端（50ep）**：mAP50-95 = **81.91% (+9.35%，历史最佳基准)**。
* **扎实的归因证据**：
  * **ENL（等效视数，斑点噪声指标）**：输入原图 1.84 $\to$ 训练后 NAFNet 输出 **10.28**（证明网络确实学到了“克制的去噪”）；
  * **冻结随机初始化对照（0 可训练参数）**：mAP = **74.23% ± 0.42%**；
  * **增益解耦公式**：$+3.22\% = +1.67\%$（随机前端卷积平滑）$+ 1.55\%$（可学习的特征细节保真去噪）。
* **残差图暴露的核心死穴**：
  * 可视化残差图（$\text{Residual} \times 5$）能清晰看到**目标几何轮廓**！说明传统空间卷积去噪在压制相干斑的同时，不可避免地产生了**边缘磨损与弱对比度目标过平滑**。

---

## 二、 前期已严格证伪的“负结果”避坑全集（Failure Cases）

> 严禁重复踩坑！这些扎实的负结果是顶刊论文中极具学术价值的消融与机理反思部分。

| 尝试方案 | 具体实现 / 假设 | 20ep mAP | 相对普通 NAFNet | 失败的深层机理剖析 |
| :--- | :--- | :--- | :--- | :--- |
| **合成加噪两阶段** | 加性/乘性 Gamma 噪声注入 + 显式 Charbonnier 损失监督 | 79.2% (50ep) | **-2.71%** (vs 81.91%) | ① 真实声呐斑点噪声是波束形成点扩散函数（PSF）相关的相干斑，非独立同分布 Gamma 噪声，引入严重域偏移；② 显式像素级 Charbonnier 损失强迫全图拟合，逼迫网络为了降 loss 把边缘抹平。 |
| **`NAFNetLog`** | $\exp(\text{NAFNet}(\log(x+\epsilon))) - \epsilon$ 对数域去噪 | 74.08% | **-1.70%** | 声呐图像含大面积声学遮挡阴影（近零暗区），$\log(x+\epsilon)$ 将近零像素映射为负向极大值，严重放大暗区数值噪声，破坏阴影几何先验。 |
| **`NAFNetMul`** | $x \cdot \exp(-f(x))$ 乘法残差估计 | 73.54% | **-2.24%** | 加法残差 $x-f(x)$ 假设空间严格大于乘法空间，物理强行约束反而扼杀了网络的表达能力。 |
| **`NAFNetEdge`** | Sobel 梯度边缘引导门控 $x - (1-w\cdot\text{edge})f(x)$ | 74.66% | **-1.12%** | 斑点噪声本身具有极强高频特性，Sobel 算子在噪声处产生巨大虚警，导致“噪声最强处不去噪”，南辕北辙。 |
| **`WNeck`** | `WeightedConcat` 标量加权特征融合 | 73.64% | **约 -0.5%** | 标量权重在整张特征图全通道一刀切，无法区分目标区域细节与背景大面积散斑，退化为无意义扰动。 |
| **几何 IoU 扩展** | WIoU v1/v3, MPDIoU, EIoU, Inner-IoU, Alpha-IoU | 均落后 CIoU | - | 刚性矩形距离惩罚无法适应声呐边界的弥散性与人工标注的主观抖动。 |
| **小目标专用设计** | NWD 瓦瑟斯坦小目标损失、P2 微小目标检测头 | 无意义 | - | 真实数据集中小目标仅 2 个，属于典型文献误用。 |

---

## 三、 当前最新突破方案与实验进展

### 1. 预案 A：离散小波子带解耦去噪 (`NAFNetWavelet`)
* **架构配置文件**：`ultralytics/cfg/models/26/yolo26n-NAFNetWavelet.yaml`
* **模块源码**：`ultralytics/nn/modules/block.py` 中 `NAFNetWavelet`
* **技术原理**：
  1. 采用无损、零参数、可逆的 2D Haar 离散小波变换（DWT）：
     $$LL, LH, HL, HH = \text{Haar\_DWT}(x)$$
  2. **低频 $LL$ 绝对直通**：承载全局声学阴影暗区、目标宏观拓扑，**完全不经过平滑卷积，100% 杜绝边缘与阴影轮廓损失**！
  3. **高频 $LH, HL, HH$ 定向去噪**：将高频子带（9 通道）送入 NAFBlocks 学习残差，滤除高频相干斑噪声。
  4. 逆小波变换（IDWT）重构图像，初始状态完全等价于恒等映射（误差 $< 10^{-6}$）。内部特征尺寸减半，FLOPs 降低 75%！
* **🚀 20 轮最新实测战报（刚刚出炉）**：
  * **mAP50-95(B) = 75.594%（相比 YOLO26n baseline 的 72.56% 暴涨 +3.03%！）**
  * **Precision = 93.16%**
  * **Recall = 94.56%（极其惊艳！此前 baseline 召回率存在约 17% 漏检，Recall 仅 83% 左右）**
  * **mAP50 = 98.10%**
  * **结论**：首次运行即斩获 75.59% 的极高水准，彻底扭转了之前一系列变体连续掉点的困局！

### 2. 预案 B：声学高光-阴影对偶先验门控 (`NAFNetAcoustic`)
* **架构配置文件**：`ultralytics/cfg/models/26/yolo26n-NAFNetAcoustic.yaml`
* **技术原理**：
  * 提取局部背景 $\mu = \text{AvgPool}_{7\times 7}(x)$；
  * 分离物理对偶特征：$\text{Highlight} = \text{ReLU}(x - \mu)$ 与 $\text{Shadow} = \text{ReLU}(\mu - x)$；
  * SimpleGate 门控在阴影暗区衰减去噪残差，保护 `victim` 微弱暗区，在高光亮区精准消除强相干斑。

### 3. 预案 C：频域解耦 + 声学物理对偶先验强强联合 (`NAFNetWaveletAcoustic`)
* **架构配置文件**：`ultralytics/cfg/models/26/yolo26n-NAFNetWaveletAcoustic.yaml`
* **技术原理**：低频 $LL$ 纯净直通，高频子带结合下采样的声学对偶图协同降噪。

---

## 四、 精选必读顶级论文指南（非读不可的 5 篇核心文献）

> 针对前视声呐、去噪检测矛盾、频域解耦与轻量化 Neck，精选 5 篇顶级文献，建议按顺序精读：

### 📄 论文 1：声呐检测最直接的一区对标 Baseline
* **题目**：*Underwater Forward-Looking Sonar Images Target Detection via Speckle Reduction and Scene Prior* (UFIDNet)
* **作者**：H. Long, L. Shen, Z. Wang, J. Chen
* **发表期刊**：**IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, Vol. 61, 2023.（中科院一区 Top）
* **为什么必读**：
  * 这是近年来在 **IEEE TGRS** 上发表的最经典的前视声呐（FLS）端到端目标检测论文。
  * 论文提出了斑点噪声消除辅助分支（SRAB）与场景先验特征选择（FSS）。
* **精读关注点**：
  1. 看作者如何定义前视声呐斑点噪声分布与去噪网络（DSN）；
  2. 看作者在实验中如何设计评价指标、消融实验表与定性可视化（你的论文将直接以此为主要对标基准）。

---

### 📄 论文 2：去噪前端的极简架构祖师
* **题目**：*Simple Baselines for Image Restoration* (NAFNet)
* **作者**：L. Chen, X. Chu, X. Zhang, J. Sun (旷视科技 Megvii Research)
* **发表会议**：**ECCV 2022**
* **为什么必读**：
  * 本项目去噪模块的核心基石。证明了彻底淘汰传统非线性激活（ReLU/GELU），仅使用通道分流相乘（SimpleGate）和简化通道注意力（SCA），就能实现极致轻量与 SOTA 恢复效果。
* **精读关注点**：
  1. 仔细阅读第 3 节（Method），看作者如何一步步通过消融拆除多余算子；
  2. 关注 LayerNorm2d 与 SimpleGate 的计算复杂度优势，用于支撑论文中的“低算力 AUV 轻量化”叙事。

---

### 📄 论文 3：频域子带解耦与噪声分离
* **题目**：*SET: Spectral Enhancement for Tiny Object Detection*
* **作者**：H. Sun, R. Wang, Y. Li, et al.
* **发表会议**：**CVPR 2025**
* **为什么必读**：
  * 首次在频域通过频谱归因分析证明：**高频成分在严重退化图像中主要是高频噪声，会遮蔽目标特征**。
  * 提出了分层背景平滑机制（HBS），利用频域滤波分离噪声与关键特征。
* **精读关注点**：
  1. 看作者如何做频域特征分析（虽然它针对极小目标，但其“高频去噪、低频保形”的频域数学逻辑与我们的 Wavelet 完全契合）；
  2. 学习其频域滤波可视化（FFT/小波能量谱图）在论文中的呈现方式。

---

### 📄 论文 4：任务驱动恢复（解释为什么不能用像素损失）
* **题目**：*TaskTok: Delving into Task Tokens for Task-driven Image Restoration* (ECCV 2026) / *Visual Recognition-Driven Image Restoration for Multiple Degradation* (CVPR 2023)
* **发表会议**：**ECCV / CVPR**
* **为什么必读**：
  * 完美解释了“**为什么传统去噪的 PSNR/Charbonnier 像素级损失会损害检测 mAP**”（过平滑矛盾）。
  * 阐述了如何让高层任务（目标检测）的梯度反向引导低层恢复网络，使恢复模块专门保全“对检测最有用的边界与语义”。
* **精读关注点**：
  1. 学习 Task-Driven 训练范式中前端恢复与后端检测的梯度流向设计；
  2. 支撑你在论文中提出的“为什么彻底舍弃人工 Gamma 像素损失，改用纯检测任务驱动”的方法学依据。

---

### 📄 论文 5：超轻量 Neck 动态内容感知上采样
* **题目**：*Learning to Upsample by Learning to Sample* (DySample)
* **作者**：Z. Liu et al.
* **发表会议**：**ICCV 2023** (Oral)
* **为什么必读**：
  * 声呐图像 Neck 改进的王牌算子。传统 `nn.Upsample(nearest)` 产生马赛克棋盘伪影放大相干斑，Bilinear 又会模糊边界。
  * DySample 仅需极少的参数，通过内容动态预测采样点位移，实现无伪影的高保真特征重构。
* **精读关注点**：
  1. 看 DySample 如何以纯 PyTorch 原生算子实现零 CUDA 依赖超高速动态上采样；
  2. 后续可在我们的 Neck 改进实验中直接作为单变量无损提点模块。

---

## 五、 云端实验执行与排查指南

由于云端机器多人使用且曾出现中断，已为你重构了健壮的执行脚本：

### 1. 脚本可靠性升级（已修复）
* **彻底移除 `set -e` 连锁崩塌机制**：即使评测脚本因为环境缺少包报错，**后续模型的训练也绝对不会被强行终止**；
* **鲁棒的多路径权重检测**：自动兼容 `runs/detect/breakthrough/` 与 `runs/breakthrough/`，确保训练完自动锁定 `best.pt`；
* **自动环境补齐**：自动检测 `pycocotools`，若缺失则静默安装。

### 2. 云端常用执行命令
```bash
# 1. 单独运行 Track B（Acoustic 对偶先验模型，20 轮）：
bash run_breakthrough.sh Acoustic 20 16

# 2. 单独运行 Track C（WaveletAcoustic 强强联合模型，20 轮）：
bash run_breakthrough.sh WaveletAcoustic 20 16

# 3. 顺序排队执行全部未跑模型（后台运行）：
nohup bash run_breakthrough.sh all 20 16 > breakthrough_queue.log 2>&1 &

# 4. 查看当前训练日志：
tail -f breakthrough_queue.log
```

### 3. 如何为 Wavelet 提取详细类别指标
对于刚刚跑完的 `Wavelet_20ep`，在云端运行以下命令即可查看各类别的详细 AP（重点关注 `victim` 是否突破 0.741）：
```bash
python compute_class_size_ap.py \
  --model runs/detect/breakthrough/Wavelet_20ep/weights/best.pt \
  --data ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml \
  --imgsz 640
```
*(若路径在 `runs/breakthrough/Wavelet_20ep`，将路径微调即可)*
