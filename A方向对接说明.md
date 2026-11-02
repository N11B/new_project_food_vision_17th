# A 方向（模型服务）对接说明

> 成员 A 交付文档 · 更新于集成前
> 面向成员 B：拿到本文件夹后，按本文档即可把模型服务接入 `app.py`，无需阅读 A 的实现细节。

---

## 1. 本文件夹交付物清单

```text
代码/
├── model_service.py                  ← 统一入口（B 只需 import 这个文件）
├── model_service_torchvision_new.py  ← 后端：torchvision >= 0.13（推荐环境使用）
├── model_service_torchvision_old.py  ← 后端：torchvision <  0.13（本机当前环境）
├── test_model_service.py             ← A 方向自测脚本（B 集成前可再跑一遍确认）
├── requirements.txt                  ← 依赖清单（已按实测环境注明版本建议）
├── foodvision_big/                   ← 模型文件（随代码本地保存，无需联网下载）
│   ├── model.py                      ← 官方模型定义（未改动）
│   ├── model.safetensors             ← 官方权重（约 30MB）
│   └── config.json                   ← 官方类别表等配置（未改动）
└── test_images/
    └── pizza_2582289.jpg             ← 真实测试图（官方示例披萨图，离线可用）
```

## 2. 快速开始（B 方向三步接入）

1. **确认环境**：能 `import torch, torchvision` 即可。把本文件夹所有文件与你的
   `app.py` 放在**同一目录**（至少 `model_service.py`、`model_service_torchvision_new.py`、
   `model_service_torchvision_old.py`、`foodvision_big/` 四项必须在 `app.py` 同目录或其可导入路径上）。
2. **导入并调用**（完整示例见第 4 节）：

   ```python
   from model_service import load_predictor, predict_topk
   ```

3. **自测**（可选但强烈建议，集成前后各跑一次）：

   ```bash
   conda activate tft-torch        # 或你自己的环境
   pip install -r requirements.txt # torch/torchvision 已有则不会被覆盖
   python test_model_service.py
   ```

   看到末行 `A 方向自测全部通过` 即可开始集成。

## 3. 冻结接口（与《02_详细代码规划方案》一致，未做任何签名改动）

```python
from model_service import load_predictor, predict_topk

predictor, device = load_predictor()
# predictor   : 官方仓库 Predictor 对象（不要直接调用它的其他方法）
# device      : "cuda" 或 "cpu"（页面直接展示即可）

predictions, seconds = predict_topk(predictor, image, top_k=5)
# image       : RGB 的 PIL.Image.Image（页面层负责 file.getvalue() -> PIL -> RGB）
# predictions : list[dict]，按 score 降序，每项 {"class_name": str, "score": float}
# seconds     : float，predictor.predict() 的耗时（秒）
```

**真实输出示例**（本机实测，测试图 `test_images/pizza_2582289.jpg`）：

```python
predictions = [
    {"class_name": "pizza",      "score": 0.9284},
    {"class_name": "bruschetta", "score": 0.0060},
    {"class_name": "lasagna",    "score": 0.0046},
    {"class_name": "garlic_bread","score": 0.0038},
    {"class_name": "omelette",   "score": 0.0032},
]
seconds ≈ 0.02（cuda）/ 0.18（cpu）
```

Top-1 = pizza（0.9284）与官方模型卡示例分数完全一致，可作为集成正确性的基准。
展示时请注意：`score` 是模型输出分数，**不是经校准的真实概率**（官方要求）。

## 4. app.py 对接示例（可直接抄）

```python
import io
import streamlit as st
from PIL import Image
from model_service import load_predictor, predict_topk

st.title("食鉴 AI：基于图像的食品分类")

@st.cache_resource          # 关键：模型只加载一次，页面交互不会反复加载
def get_model():
    return load_predictor()

uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
photo = st.camera_input("或拍照")          # 二选一控件建议用 radio 切换
source = uploaded or photo

if source is not None and st.button("识别食品"):
    try:
        image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")
        predictor, device = get_model()
        predictions, seconds = predict_topk(predictor, image, top_k=5)
        st.write(f"设备：{device}　耗时：{seconds:.3f} 秒")
        st.write(f"Top-1：{predictions[0]['class_name']}（{predictions[0]['score']:.4f}）")
        st.table(predictions)              # Top-5
    except Exception as exc:
        st.error(f"识别失败：{exc}")        # 不要用 except: pass，不要造假数据
```

要点复述：
- `load_predictor()` 只应出现在 `@st.cache_resource` 函数内部，调用一次；
- 没有图片时不调用模型；异常要 `st.error()` 展示，不得用固定结果冒充预测；
- `app.py` 内不要写任何模型加载细节，全部通过上面两个函数。

## 5. 环境需求与 torchvision 版本说明（重要）

| 项 | 要求 |
| --- | --- |
| Python | 3.9 ~ 3.11（本机实测 3.9.16） |
| torch / torchvision | 能正常导入即可，本项目不固定、不重装；有 CUDA 环境则自动用 GPU |
| 其余依赖 | `pip install -r requirements.txt`（streamlit 1.39.0、Pillow 10.4.0、huggingface_hub 0.26.2、safetensors 0.4.5） |
| 网络 | 模型文件已本地化，**运行不联网**；仅当 `foodvision_big/` 缺失时才尝试自动下载 |

**torchvision 版本与两个后端脚本**：

- **推荐：torchvision >= 0.13**。官方 `model.py` 可以原样加载，走
  `model_service_torchvision_new.py`，无任何兼容处理。新环境搭建时请优先满足此条。
- **不推荐但可用：torchvision < 0.13**（本机当前为 0.11.1）。该版本没有 Weights 枚举，
  统一入口会自动切换到 `model_service_torchvision_old.py`，其中内置了与官方模型卡
  语义完全一致的兼容层（resize 288 bicubic → center crop 288 → ImageNet 归一化），
  官方 `model.py` 文件保持原样未改。
- **无需手工选择**：`model_service.py` 会读取 `torchvision.__version__` 自动分派；
  两个后端的 `load_predictor()` 接口完全相同，B 的代码不用感知差异。
- 判断当前用了哪个后端：读 `model_service.BACKEND`（加载后为实际后端模块名），
  自测脚本第 2 步也会打印。

## 6. 模型文件与离线交接

- 三个官方文件已保存在 `代码/foodvision_big/`，**整个代码文件夹可直接压缩拷贝交接**，
  对方解压后即可离线运行，无需 Hugging Face 账号或网络。
- 若该目录缺失（如只拷了 .py 文件），`load_predictor()` 会自动下载：
  先官方站 huggingface.co，失败则自动切国内镜像 hf-mirror.com（脚本已内置），
  也可手动从 `https://hf-mirror.com/shalev396/foodvision-big` 下载
  `model.py / model.safetensors / config.json` 放入该目录。
- 模型来源与许可：`shalev396/foodvision-big`，MIT 许可证；模型仅覆盖 Food-101 的
  101 类菜品，范围外的输入也会被强行归到某一类（无"其他"类），README 展示时需说明。

## 7. 设备选择与显存保护（本机踩坑记录，务必了解）

- **显存需求极小**：模型只有 7.8M 参数，单张 288×288 图，显存占用几百 MB 以内。
- **自动选择设备**：`load_predictor()` 会先用**一次性子进程**做 CUDA 烟雾测试
  （真实加载并推理一次），成功则主进程用 cuda，失败则主进程直接用 cpu。
  这样即使 CUDA 环境"看着可用但实际初始化失败"（显存被占满、cuDNN 算法失败等），
  也不会影响主进程。
- **踩坑原因**：实测发现 cuDNN 初始化失败（`fatal: Memory allocation failure`）会
  污染当前进程，之后连 CPU 的 0.3MB 分配都会失败。因此**绝不能**在主进程里
  "先试 CUDA 失败再退 CPU"，必须子进程试错——这就是 `model_service.py` 里
  `_cuda_smoke_test()` 的由来。
- **强制指定设备**：设置环境变量 `FOODVISION_DEVICE=cpu`（或 `cuda`）可跳过探测。
  演示当天如遇 GPU 被其他程序占用，`set FOODVISION_DEVICE=cpu` 后再启动 streamlit 即可。
- 集成验收时页面显示的 `device` 就是这个机制的结果；CUDA 与 CPU 的推理结果完全一致。

## 8. 与原规划的差异（集成时请对照）

| 原规划 | 实际交付 | 原因 |
| --- | --- | --- |
| 模型下载到 HF 默认缓存目录 | 模型三文件落地到 `代码/foodvision_big/`，随代码保存 | 便于离线交接与答辩演示（A 方向要求） |
| `model_service.py` 单文件 | `model_service.py` + 两个 torchvision 版本后端脚本 | B 环境未知，双版本保证任何 torchvision 环境都能跑；统一入口接口不变，B 无需改代码 |
| 加载时直接选 `cuda` if available | 子进程 CUDA 烟雾测试 + 预热 + 自动回退 CPU，支持 `FOODVISION_DEVICE` 覆盖 | 本机实测 cuDNN 偶发失败且污染进程，见第 7 节 |
| `model_service.py` 完全不改官方文件 | 仍未改；新增兼容层只在 torchvision<0.13 时注入 | 兼容旧版 torchvision |
| 其余（函数签名、返回格式、异常上抛、不做预处理/训练） | **完全一致，无差异** | — |

## 9. A 方向测试记录（2026-10-19 实测）

- 环境：Windows 11，conda env `tft-torch`（Python 3.9.16，torch 1.10.0+cu113，
  torchvision 0.11.1，RTX 3060 Laptop GPU）。
- 命令：`python test_model_service.py`（默认）与 `FOODVISION_DEVICE=cpu python test_model_service.py`。
- 结果：两者均输出 `A 方向自测全部通过`；
  - cuda 可用时设备=cuda，Top-5 推理约 0.02 秒；
  - GPU 异常/被占用时自动回退 cpu，约 0.18 秒；
  - Top-1 pizza 0.9284，与官方模型卡基准一致；
  - 校验项（条数、降序、字段、耗时格式）全部通过。
- 未做全链路测试（`streamlit run app.py` 属于 B 方向联调范围）。

## 10. 集成常见问题排查（按顺序查）

1. `python test_model_service.py` 是否通过？不通过先解决 A 侧环境，再谈集成。
2. `app.py` 与 `model_service*.py`、`foodvision_big/` 是否同目录？
3. 页面反复变慢/重新下载？→ 检查 `@st.cache_resource` 是否包住了 `load_predictor()`。
4. 页面显示 cpu 但预期 cuda？→ 属于第 7 节的自动回退，检查显存占用或强制 `FOODVISION_DEVICE`。
5. 首次运行有 UserWarning "兼容层"？→ 正常现象（torchvision<0.13），不影响结果。
6. 仍失败 → 把终端完整报错发给成员 A。
