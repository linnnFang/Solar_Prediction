# 开发日志 / Journal

## 2026-07-03 — 数据重建 + 可复用模块化

### 数据装配
- 发现旧的 `GEFCom2014_Zone*_Assembled.csv` 有问题：缺 2014 年 5–6 月、每 zone 约 2176 个空洞，`_New` 文件还重复膨胀 5.6 倍。
- 改用 **Task 15** 三个源文件重建：`train15`（历史真值）+ `Solution`（预测月真值）+ `predictors15`（天气）。
- 结果每 zone **19,704 行，连续无缺、无重复**（旧的只有 16,061 行带空洞）。

### 累积量 / 时区
- 4 个累积字段（SSRD/STRD/TSR/TP = VAR169/175/178/228）按预报日去累积，新增 `*_dea` 每小时真实值列，保留原始。
- 确认时间戳是 UTC；数据反推本地 = **UTC+10**（澳洲东部），写进 `config.yaml` 的 `tz_offset`。

### 新增可复用模块（`src/helper/`）
- `data_preprocessing.py` — `GEFComTask15` 装配类（load / deaccumulate / to_processed）；保留 `load_all_zones()` 兼容。
- `features.py` — `FeatureBuilder` 链式特征：本地时间、cyclic（hour/doy 的 sin·cos）、风速；太阳几何/晴空留占位。
- `backtest.py` — `WalkForward` 滚动回测：lookback/horizon 可调，`gefcom_monthly()` 复现官方 14 个月度折。
- `metrics.py` — `1−RMSE/Cap`、`1−MAE/Cap`，`monthly_accuracy`（日→月平均、仅白天）。
- `models.py` — `BaseForecaster` 接口 + `SklearnForecaster` + `build_model` 工厂（linear/rf/xgboost/lightgbm，懒加载）；`fit(**fit_params)` 透传早停。
- `params.py` + `configs/model_params.yaml` — 模型超参集中管理，`get_params("预设")`，调参只改 yaml。

### Notebook
- `DEFCom_EDA.ipynb` — 改用新装配；数据质量复查（0 缺失/0 空洞）、本地时间日内曲线（夜里不发电）、去累积对比、白天 vs 全天准确率、落盘、回测切分预览。
- `baseline_XGBoost.ipynb` — 模型走 `build_model` + 参数走 `get_params`；新增第 6 节 XGBoost / LightGBM / RandomForest 三模型对比。

### 产物 / 环境
- 输出干净基表 `data/process/gefcom_task15_clean.parquet`（59,112 行 × 19 列）。
- 安装 lightgbm 4.6.0。

### 关键结果
- 主 XGBoost：test RMSE 0.1021，相对 climatology 提升 16.1%（数据变干净后从 8.8% 提升）。
- 三模型对比：rf 0.0998 / xgboost 0.1003 / lightgbm 0.1008（相当接近）。

### 未做（留待以后）
- 太阳几何 / 晴空指数（需站点纬度 + pvlib）。
- 短临 / 超短期（15min、4h）指标 → 属于 SKIPPD 天空图像数据集。
- 深度学习 / 序列模型（接口已预留）。

## 2026-07-22 — Transformer「训练极慢」排查：OpenMP 死锁

### 现象
- `05_NationalRenewable_baseline.ipynb` 的 Transformer 单元跑 20+ 分钟不出结果，疑似 MPS 没生效。

### 排查过程（两个独立问题）
1. **MPS 本身没问题**：`device="mps"` 一路传到 `Trainer`，实测 MPS 比 CPU 快 8 倍（25.7 vs 197.6 ms/batch）。
   - 顺带排除若干猜想：每 batch `loss.item()` 同步开销可忽略（23.3 vs 22.4 ms）；加大 batch 不提吞吐（恒定 ~11k windows/s，模型太小）；`num_workers=4` 反而比 `0` **慢 4 倍**（macOS 多进程 DataLoader 的 spawn/IPC 开销），故保持 `num_workers=0`。
2. **机器内存耗尽**（真实但非主因）：swap 31 GB 用满 97%、11 GB 压缩内存、真实空闲仅 63 MB、14 个残留 Jupyter kernel。整机换页导致全面变慢。重启解决。
3. **根因：OpenMP 运行时冲突死锁**。重启后仍复现，且 kernel CPU 时间冻结、`%CPU=0`——是**卡死不是慢**。`/usr/bin/sample` 抓栈定位到：
   `torch.sin → sin_kernel → TensorIterator::for_each → _pthread_cond_wait`
   即 `PositionalEncoding` 里的 `torch.sin`（建模型时执行）在 CPU 线程池上死等。

### 根因说明
环境里有 **4 份互相冲突的 OpenMP 运行时**（pip wheel 各自捆绑）：
`torch/lib/libomp.dylib`、`sklearn/.dylibs/libomp.dylib`、`skimage/.dylibs/libomp.dylib`、`cvxopt/.dylibs/libgomp.1.dylib`（GNU）。
notebook 里 **cell 14 先跑 XGBoost/sklearn 加载了一份，cell 18 才 import torch**，到 cell 20 第一次 CPU 并行运算即死锁/段错误。

### 修复方案（实测）
| 方案 | 结果 |
|---|---|
| `OMP_NUM_THREADS=1` | ✅ |
| `torch.set_num_threads(1)` | ✅ **采用** |
| 先 import torch 再 sklearn | ✅ |
| `KMP_DUPLICATE_LIB_OK=TRUE` | ❌ 仍段错误（常见偏方，无效） |

训练跑在 MPS 上，CPU 线程数不影响速度，故限为 1 线程零代价。

### 改动
- `src/vanilla_transformer/__init__.py` — 包级兜底：检测到 MPS/CUDA 时自动 `set_num_threads(1)`；纯 CPU 机器保持默认以免拖慢 CPU 训练；可用 `SOLAR_TORCH_THREADS` 覆盖。
- `src/vanilla_transformer/trainer.py` — `fit()` 每轮打印耗时（`epoch 1 | 3.5s | train … | val …`），便于区分「慢」与「卡死」。
- `05_NationalRenewable_baseline.ipynb` — cell 18 显式加 `torch.set_num_threads(1)` + 注释。
- `03_GEFCom_patchTST.ipynb` / `patchTST.ipynb` — 原本靠「torch 先于 sklearn 导入」碰巧安全，改为显式防护，不再依赖导入顺序。

### 结果
- 全 notebook 端到端 **66.5 秒**跑通（此前无限卡死），Transformer 训练 55.6 秒、每轮稳定 **2.7 秒**，best val loss 0.00105。
- 产物：`notebook/model/05_NationalRenewable_baseline_RESULTS.ipynb`（带完整输出）。
- 站点 1 测试集对比（35,040 个预测，同一批目标时刻）：

| 模型 | acc_mae | acc_rmse |
|---|---|---|
| XGBoost | **0.9846** | 0.9642 |
| vanilla_transformer | 0.9837 | **0.9646** |
| persistence | 0.9787 | 0.9571 |

两个模型均明显超过 persistence；XGBoost 的 MAE 略优，Transformer 的 RMSE 略优，基本打平。

### 排查手法备忘
- **区分「慢」与「卡死」**：连续观察 `ps -o time` 累计 CPU 时间，冻结不动 = 阻塞而非计算慢。
- **抓栈**：`/usr/bin/sample <pid>`（conda 会遮蔽 `sample`，须用绝对路径；`py-spy` 在 macOS 需 root）。
- 用 `nbclient` 逐 cell 执行并打印耗时，可精确定位卡在哪个 cell（`nbconvert` 只在全部跑完才写文件，看不到进度）。

### 待办
- 环境层面根治（可选）：统一到单一 OpenMP（如 conda-forge 的 `llvm-openmp`），或移除 `cvxopt` 的 GNU `libgomp`。当前是代码层规避。
- 定期清理残留 Jupyter kernel，避免再次耗尽内存。

## 2026-07-22 — 统一结果报告：三条管线的口径对齐

新建 `notebook/model/06_Unified_Result.ipynb`（英文，30 cells，端到端 66 秒跑通）。
目标不是再训一遍模型，而是把 **GEFCom / SKIPP'D / Renewable** 三条管线的
**schema、滑动窗口、指标口径**摆在一起，讲清楚哪些数字能比、哪些不能。

### 报告结构
- §1 三个数据集逐列 schema（`dtype / non_null / n_unique / min / max / mean`，目标列标 TARGET）+ ECMWF 变量字典
- §2 滑动窗口口径（按 pipeline 分表）+ 真实窗口数统计 + 示意图
- §3 归一化，拆成三个层次：① 目标量纲 ② 输入缩放 ③ 指标归一化
- §4 全部指标汇总，**每行标注 unit / normalized_by / source**
- §5 可比性分组 + 各数据集内部最佳

### 关键设计决定
- **静态表一律用 markdown + LaTeX，不用 `pd.DataFrame` 构造**。窗口表、归一化表、变量字典本就没有计算，用 DataFrame 只会让源码变成一大坨 dict 字面量。注意 Jupyter 的 markdown **不渲染 LaTeX `tabular`**（MathJax 只处理数学环境），所以表格骨架用 markdown 表、公式用真 LaTeX；将来 `nbconvert` 导出 PDF 会自动转成真 LaTeX 表格。
- **指标来源逐行标注** `recomputed` / `artifact` / `not run`：便宜的模型（GEFCom XGBoost 1.5s、Renewable 全套）现场重算保证与代码一致；SKIPP'D 读 `artifacts/skippd/comparison.csv`。
- **GEFCom 两个深度模型填 `NaN` + `not run`，不是漏跑也不是失败**。$L=168,H=24$ 训练约 15~30 分钟，放进报告会让每次运行从 90 秒变成一小时。三选一里：不放进表 → 会忘记它们存在；抄旧数字 → 等于伪造且代码改了也不变；**填 NaN 标 not run** → 可见但不可能被误读。开关 `RUN_GEFCOM_NEURAL`。

### 澄清的三个口径问题（都查了代码，非印象）
1. **`POWER` 最大值是 1.004 不是 1.0** —— 全表仅 1 行越界（`2013-11-04 02:00`, ZONE 3, 1.00355），GEFCom 原始数据已知瑕疵。schema 表读原始文件故如实显示；建模前一律 `clip(0,1)`。
2. **12 个 NWP 特征是原始 ECMWF 物理量，完全未归一化**，跨约 6 个数量级（VAR79 均值 0.017 vs VAR134 均值 93810）。这正是 Transformer/PatchTST 必须 z-score 的原因；XGBoost 是树模型故不需要。VAR157 可 >100（NWP 允许过饱和）、VAR165/166 为负（风向），均非异常。
3. **「白天过滤」三条管线各不相同**（此前报告里写错过，已改）：
   - GEFCom：`is_daytime = VAR169_dea > 0`，判据是**辐照度**而非功率是否为 0（`src/helper/features.py:35`）
   - SKIPP'D：evaluator 里 `is_daytime` **硬编码 True，不过滤**，因数据集本身已剔除夜间（`src/baseline_skippd/evaluator.py:39`）
   - Renewable：`monthly_accuracy_15min` **完全不过滤**，夜间零值全部计入

### 新发现：Renewable 的分数有相当部分来自夜间零值
测试集 **50.1% 的点目标恰为 0**。全时段是本任务要求的考核口径，主表保持不变，但新增诊断表并列两种口径：

| 模型 | acc_rmse 全时段 | acc_rmse 仅白天 | 差值 |
|---|---|---|---|
| persistence | 0.9571 | 0.9390 | 0.0181 |
| xgboost | 0.9642 | 0.9495 | 0.0147 |
| vanilla_transformer | 0.9646 | 0.9501 | 0.0145 |

即约 1.5~1.8 个百分点是「夜里预测 0」白送的。呼应 05 notebook next steps 里的提醒。

### 两个 monthly accuracy 函数口径不同，不可互比
- `monthly_accuracy`（GEFCom/SKIPP'D）：先算**每日**准确率，再按天平均到月
- `monthly_accuracy_15min`（Renewable）：把**整月所有 15 分钟点**汇总起来算一次

### 各数据集内部最佳（跨行不可比）
跨数据集绝对不可比：量纲不同（无量纲 / kW / MW）、容量差约 1600 倍（30.1 kW vs 50 MW）、
分辨率与预测步长不同（1h/1min/15min，预测 24/1/1 步）、指标定义不同。

### 补充轮：GEFCom 数字补齐 + 口径纠错（同日）

**① GEFCom 深度模型改为从 02/03 的已保存输出自动解析，不再训练也不再是 NaN。**
写了个解析器扫 notebook 的 outputs 抓 `RMSE=x MAE=y`。相比硬编码数字：可追溯到具体文件、
重跑 02/03 后自动更新、解析不到会直接抛错而不是静默留空。取到：

| 来源 | 模型 | RMSE | MAE |
|---|---|---|---|
| 02 | Vanilla Transformer | 0.1103 | 0.0619 |
| 02 / 03 | Seasonal-naive | 0.1356 | 0.0600 |
| 03 | PatchTST (history) | 0.1134 | 0.0587 |
| 03 | PatchTST + future NWP | 0.0757 | 0.0383 |

**② 发现并修正了一个会误导人的排名错误。** 原「各数据集内部最佳」按 dataset 分组，
会得出「XGBoost 0.1021 打败 Transformer 0.1103」——**这是错的，它们不是同一个任务**：
XGBoost 是逐行表格模型、用目标时刻 NWP 预测**单步**；Transformer/PatchTST 是
$L=168 \to H=24$ 的**多步纯历史**预测。改为按**可比任务组**排名：

| 可比组 | 最佳 | 值 | 组内模型数 |
|---|---|---|---|
| GEFCom / 单步 + 目标时刻 NWP | XGBoost | RMSE 0.1021 | 2 |
| GEFCom / 168→24 多步纯历史 | Vanilla Transformer | RMSE 0.1103 | 3 |
| GEFCom / 168→24 用未来 NWP（更容易） | PatchTST + future NWP | RMSE 0.0757 | 1 |
| SKIPP'D | vanilla_transformer | acc_rmse 0.8933 | 5 |
| Renewable | vanilla_transformer | acc_rmse 0.9646 | 3 |

分组后的正确结论：多步纯历史组里 Transformer 与 PatchTST **都超过了该组的正确基线
seasonal-naive（0.1356）**，分别提升 18.7% 和 16.4%。`PatchTST + future NWP` 的 0.0757
之所以显著更低，是因为它能看到**目标时刻的未来气象预报**，属于另一个更容易的任务设定，
必须单独成组，不可与 history-only 模型并列。

**③ 新增 `acc` 与 `RMSE` 的恒等式验证 cell。** 关系是
$\mathrm{acc} = 1 - \dfrac{\text{error}}{P_{\text{cap}}}$，**只有** $P_{\text{cap}}=1$ 时
才退化成 $1-\mathrm{RMSE}$——这对 GEFCom 成立（`POWER` 已被数据集除过容量），
对 SKIPP'D（30.1 kW）和 Renewable（50 MW）不成立。且因容量在站内是常数，
$\operatorname{mean}_i(1 - e_i/P_{\text{cap}}) = 1 - \operatorname{mean}_i(e_i)/P_{\text{cap}}$，
所以月均之后恒等式**依然精确成立**。报告里逐行程序验证，全部 `OK`。

**④ 记录 `doy_sin` / `doy_cos` / `tod_sin` / `tod_cos` 的含义**（`src/national_renewable.py:132-136`）：
周期性时间的圆周编码，$\sin(2\pi \cdot \mathrm{doy}/365.25)$、$\cos(\cdot)$，`tod` 除以 1440 分钟。
目的是抹平跨年/跨日接缝——直接用 `doy` 会让 12月31日（365）与 1月1日（1）相差 364，
编码后两者距离 0.0215，与相邻两天的 0.0172 几乎相等。**必须 sin/cos 成对**：单用 sin 有歧义
（2月15日 sin=+0.7113 与 5月17日 sin=+0.7067 几乎撞车，靠 cos 符号相反才分得开）。
`doy_sin` 的 `mean ≈ 1.3e-05` 接近 0 是**数据完整覆盖整年**的好信号，明显偏离 0 则说明年周期不完整。

### 待办
- 可考虑把「仅白天」诊断也加到 SKIPP'D 与 GEFCom，形成统一的双口径报告。
- GEFCom 若重训 02/03，报告会自动读到新值，无需改 06。
