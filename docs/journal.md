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
