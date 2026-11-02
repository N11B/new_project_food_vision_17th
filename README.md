# 食鉴 AI 网页版

这是原 Streamlit 食物识别作业的 Vercel 部署版。项目将 FoodVision Big 的
PyTorch 权重转换为 ONNX，并使用 ONNX Runtime Web 在用户浏览器中完成推理。

## 架构

```text
上传或拍摄图片
      ↓
浏览器 Canvas 完成裁剪、缩放和 ImageNet 标准化
      ↓
ONNX Runtime Web 运行 EfficientNet-B2
      ↓
页面显示 Top-1、Top-5 和本地热量参考值
```

此版本没有后端 API：图片不会上传到服务器，Vercel 只负责托管 HTML、CSS、
JavaScript、WASM 运行时、约 30 MB 的 ONNX 模型和本地 JSON 营养表。这样可以避开
Streamlit 与 Vercel 架构不兼容、PyTorch 函数体积较大和服务器冷启动等问题。

## 本地热量数据库

- `public/data/nutrition.json` 覆盖模型的全部 101 个 Food-101 类别。
- 每个类别包含中文名称、每 100 g 典型热量和配方波动系数。
- 页面自动显示每 100 g 参考值、参考区间，并可按用户输入的克数线性换算。
- 构建前会运行 `scripts/validate-nutrition.mjs`，检查数据库与模型类别是否一一对应；
  缺失、冗余或字段无效都会使 Vercel 构建失败，避免上线不完整数据。
- 参考体系为 [USDA FoodData Central](https://fdc.nal.usda.gov/)。数值是课程演示用的
  类别级典型熟制菜品近似值；Food-101 标签没有配方、份量和烹饪用油信息，因此不能
  视为对照片中食物的精确营养检测。

## 本地运行

需要 Node.js 18 或更高版本。

```bash
npm install
npm run dev
```

浏览器打开终端显示的本地地址。首次识别前需要加载约 30 MB 模型；浏览器缓存后
再次访问会更快。

生产构建：

```bash
npm run build
npm run preview
```

构建结果位于 `dist/`。

## GitHub 和 Vercel 发布

1. 将本目录中的改动提交并推送到 GitHub。
2. 登录 Vercel，选择 **Add New → Project**。
3. 导入该 GitHub 仓库。
4. 当前 Git 仓库的网页工程位于 `代码` 子目录，因此将 **Root Directory** 设置为
   `代码`。如果以后把网页文件移动到仓库根目录，再改回仓库根目录即可。
5. Vercel 会读取 `package.json` 和 `vercel.json`，执行 `npm run build`，并发布
   `dist/`。
6. 以后向生产分支（通常为 `main`）推送提交，Vercel 会自动重新构建和发布。

无需配置 Python、GPU、环境变量或 API 密钥。

## 关键文件

- `index.html`：网页结构。
- `src/main.js`：图片预处理、ONNX 推理和交互逻辑。
- `src/style.css`：响应式页面样式。
- `public/model/foodvision-big.onnx`：浏览器推理模型。
- `public/model/classes.json`：Food-101 类别顺序。
- `public/data/nutrition.json`：101 类离线热量参考数据库。
- `scripts/validate-nutrition.mjs`：构建前的数据完整性检查。
- `DATA_SOURCES.md`：热量数据口径、计算方法和使用限制。
- `vercel.json`：Vercel 构建输出及模型缓存策略。
- `export_onnx.py`：开发时从原 PyTorch 权重重新导出 ONNX；Vercel 不运行它。

原来的 `app.py`、`model_service*.py`、`foodvision_big/` 和 `requirements.txt` 保留为
PyTorch/Streamlit 参考实现，不参与 Vercel 构建。

## 模型一致性

导出时使用原项目的 `tft-torch` 环境和本地权重，并用
`test_images/pizza_2582289.jpg` 对比 PyTorch 与 ONNX：

- PyTorch Top-1：`pizza`
- ONNX Top-1：`pizza`
- ONNX 分数：约 `0.928385`
- 两端概率最大绝对差：约 `1.2e-7`

网页预处理遵循原模型的规则：中央正方形裁剪、缩放至 288×288、转换为 RGB，
再使用 ImageNet mean/std 标准化。浏览器 Canvas 与 Pillow 的重采样实现不同，像素
级结果可能存在极小差异。

## 模型来源与限制

- 模型：[`shalev396/foodvision-big`](https://huggingface.co/shalev396/foodvision-big)
- 结构：EfficientNet-B2
- 类别：Food-101 的 101 类菜品
- 许可：MIT

模型分数不是经过校准的真实概率。非食物图片和 Food-101 范围外的食物仍会被归入
某个已知类别。本项目用于课程演示，不应作为营养、过敏或食品安全判断依据。

## 重新导出 ONNX

仅在 PyTorch 权重发生变化时需要重新导出。使用已经安装 PyTorch、TorchVision、
ONNX 和 ONNX Runtime 的现有 Python 环境运行：

```bash
python export_onnx.py
```

脚本会覆盖 `public/model/foodvision-big.onnx` 和 `classes.json`，并自动检查模型、
比较 PyTorch 与 ONNX 输出。
