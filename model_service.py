# -*- coding: utf-8 -*-
"""model_service.py —— 成员 A 负责：模型服务与运行环境（统一入口）。

冻结接口（成员 B 的 app.py 只依赖这两个函数，不得更改签名与返回格式）：

    predictor, device_name = load_predictor()
    predictions, seconds = predict_topk(predictor, image, top_k=5)

模型来源：https://huggingface.co/shalev396/foodvision-big （MIT 许可证）

模型文件随代码保存在 foodvision_big/ 子目录中（model.py / model.safetensors /
config.json，约 30MB），加载时优先使用本地文件、不联网；仅当文件缺失时才尝试自动
下载（先官方站后国内镜像 hf-mirror.com），便于直接拷贝交接。

torchvision 兼容性（自动选择后端脚本，无需手工指定）：
- torchvision >= 0.13（推荐）：使用 model_service_torchvision_new.py，
  直接加载官方 model.py；
- torchvision <  0.13（本机当前环境）：使用 model_service_torchvision_old.py，
  内置等效兼容层（预处理与官方模型卡完全一致）。
两个后端脚本接口完全相同，详见交接文档《A方向对接说明.md》。

显存策略：模型仅 7.8M 参数，显存占用很小。加载后会用一张 288x288 假图做一次
预热推理，若 CUDA 推理失败（显存不足 / cuDNN 不可用），自动回退到 CPU 运行，
保证演示稳定；也可用环境变量 FOODVISION_DEVICE=cpu 强制使用 CPU。
"""

import importlib
import importlib.util
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量配置
# ---------------------------------------------------------------------------

# Hugging Face 模型仓库 ID
MODEL_REPO_ID = "shalev396/foodvision-big"

# 模型文件落地目录：与本文件同级的 foodvision_big 文件夹（随代码一起保存，可直接拷贝）
MODEL_DIR = Path(__file__).resolve().parent / "foodvision_big"

# 推理必需的仓库文件
REQUIRED_FILES = ("model.py", "model.safetensors", "config.json")

# 国内镜像（直连 huggingface.co 失败时自动回退）
MIRROR_ENDPOINT = "https://hf-mirror.com"

# 子进程下载超时（秒）。模型约 30MB，正常几分钟内完成。
DOWNLOAD_TIMEOUT_SECONDS = 900

# 当前实际使用的后端脚本名（load_predictor() 调用后更新，供测试/交接记录用）
BACKEND = "(未加载)"

# 在子进程中执行的下载脚本：把仓库文件直接落到 MODEL_DIR（不使用符号链接缓存）
_DOWNLOAD_SNIPPET = (
    "import sys\n"
    "from huggingface_hub import snapshot_download\n"
    "snapshot_download(\n"
    "    repo_id='shalev396/foodvision-big',\n"
    "    local_dir=sys.argv[1],\n"
    "    allow_patterns=['model.py', 'model.safetensors', 'config.json'],\n"
    ")\n"
    "print('download done')\n"
)


# ---------------------------------------------------------------------------
# 模型下载与就绪检查
# ---------------------------------------------------------------------------

def is_model_downloaded():
    """模型三个必需文件是否都已存在于本地目录。"""
    return all((MODEL_DIR / name).is_file() for name in REQUIRED_FILES)


def _endpoint_reachable(endpoint):
    """快速探测一个 HF 端点是否可达（10 秒超时）。"""
    import urllib.request

    url = "{}/api/models/{}".format(endpoint.rstrip("/"), MODEL_REPO_ID)
    try:
        with urllib.request.urlopen(url, timeout=10):
            return True
    except Exception:
        return False


def _download_model_once(endpoint):
    """在子进程中用指定端点下载模型文件。

    使用子进程 + 环境变量的原因：huggingface_hub 在导入时读取 HF_ENDPOINT，
    在同一进程内切换端点不可靠；子进程方式可以干净地分别尝试官方站与镜像站。
    """
    env = os.environ.copy()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    result = subprocess.run(
        [sys.executable, "-c", _DOWNLOAD_SNIPPET, str(MODEL_DIR)],
        env=env,
        capture_output=True,
        text=True,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "模型下载失败（端点：{}）。\nstdout: {}\nstderr: {}".format(
                endpoint or "默认(huggingface.co)", result.stdout, result.stderr
            )
        )


def ensure_model_downloaded():
    """确保模型文件在 MODEL_DIR 就绪；返回本地模型目录字符串。

    尝试顺序：
    1. 本地文件已齐全 -> 直接返回，不做任何网络请求；
    2. HF_ENDPOINT 环境变量指定的端点（未设置则为官方 huggingface.co）；
    3. 国内镜像 hf-mirror.com。
    全部失败则抛出 RuntimeError，由调用方向用户展示。
    """
    if is_model_downloaded():
        return str(MODEL_DIR)

    candidates = [os.environ.get("HF_ENDPOINT") or "https://huggingface.co"]
    if MIRROR_ENDPOINT not in candidates:
        candidates.append(MIRROR_ENDPOINT)

    errors = []
    for endpoint in candidates:
        if not _endpoint_reachable(endpoint):
            errors.append("端点不可达：{}".format(endpoint))
            continue
        try:
            _download_model_once(endpoint)
            if is_model_downloaded():
                return str(MODEL_DIR)
            errors.append("端点 {} 下载后文件不齐全".format(endpoint))
        except Exception as exc:  # 记录后继续尝试下一个端点
            errors.append("端点 {} 下载异常：{}".format(endpoint, exc))

    raise RuntimeError(
        "模型下载失败，已尝试的端点：\n" + "\n".join(errors)
        + "\n请检查网络，或手动从 https://hf-mirror.com/shalev396/foodvision-big "
        "下载 model.py / model.safetensors / config.json 三个文件，放入：{}".format(MODEL_DIR)
    )


# ---------------------------------------------------------------------------
# 官方 model.py 导入与加载（供两个后端脚本复用）
# ---------------------------------------------------------------------------

def import_repo_model(model_dir):
    """从模型目录加载官方 model.py 模块（不污染 sys.path，避免同名冲突）。"""
    model_py = Path(model_dir) / "model.py"
    if not model_py.is_file():
        raise RuntimeError("模型目录中缺少 model.py：{}".format(model_py))
    spec = importlib.util.spec_from_file_location("foodvision_repo_model", model_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cuda_smoke_test(timeout_seconds=180):
    """在一次性子进程中测试 CUDA 推理是否真实可用，返回 True/False。

    为什么不用主进程直接试：cuDNN 初始化失败（显存不足/驱动问题）会污染当前
    进程的内存状态，后续连 CPU 分配都可能失败（已实测复现）。因此先用一次性
    子进程试错，主进程再根据结果直接选择设备，全程干净。
    """
    code = (
        "import os, sys\n"
        "os.environ['FOODVISION_DEVICE'] = 'cuda'\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import model_service\n"
        "predictor, device = model_service.load_predictor()\n"
        "print('SMOKE_OK ' + device)\n"
    )
    env = os.environ.copy()
    env["FOODVISION_DEVICE"] = "cuda"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(Path(__file__).resolve().parent)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and "SMOKE_OK" in (result.stdout or "")


def load_repo_predictor_with_fallback(repo_model, model_dir):
    """加载模型并做一次预热推理；CUDA 不可用时安全回退 CPU。

    设备选择顺序：
    1. 环境变量 FOODVISION_DEVICE=cpu/cuda 强制指定；
    2. 否则先用一次性子进程探测 CUDA 真实可用性，可用则用 cuda，否则直接 cpu。

    返回 (predictor, device_name)。
    """
    import torch
    from PIL import Image

    forced = os.environ.get("FOODVISION_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        candidates = [forced]
    else:
        candidates = ["cuda", "cpu"] if _cuda_smoke_test() else ["cpu"]

    last_error = None
    for device in candidates:
        try:
            predictor = repo_model.load(str(model_dir), device=device)
            # 预热：用最小假图推理一次，提前暴露潜在问题
            predictor.predict(Image.new("RGB", (288, 288)), top_k=1)
            return predictor, device
        except Exception as exc:
            last_error = exc
            if device == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(
        "模型加载/预热失败（尝试过的设备：{}）。最后错误：{}".format(candidates, last_error)
    )


# ---------------------------------------------------------------------------
# 冻结接口
# ---------------------------------------------------------------------------

def _torchvision_version():
    """返回 (major, minor)，兼容 '0.11.1+cpu' 这类带后缀的版本号。"""
    import torchvision

    match = re.match(r"(\d+)\.(\d+)", torchvision.__version__)
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def load_predictor():
    """加载 FoodVision Big 模型（根据 torchvision 版本自动选择后端脚本）。

    返回：
        (predictor, device_name)
        predictor   : 官方仓库 Predictor 对象，用 predict_topk() 调用；
        device_name : "cuda" 或 "cpu"（CUDA 不可用或预热失败时自动回退 CPU）。
    """
    global BACKEND
    if _torchvision_version() >= (0, 13):
        backend_name = "model_service_torchvision_new"  # 推荐
    else:
        backend_name = "model_service_torchvision_old"  # 本机当前环境
    backend = importlib.import_module(backend_name)
    BACKEND = backend_name
    return backend.load_predictor()


def predict_topk(predictor, image, top_k=5):
    """用加载好的 predictor 对一张图片做 Top-K 预测。

    参数：
        predictor : load_predictor() 返回的第一个值；
        image     : RGB 格式的 PIL.Image.Image（由页面层负责转换）；
        top_k     : 返回类别数，默认 5。

    返回：
        (predictions, inference_seconds)
        predictions 为按 score 降序的列表，例如：
            [{"class_name": "pizza", "score": 0.92}, ...]
        inference_seconds 为 predictor.predict() 的耗时（秒）。

    说明：score 是模型输出分数，不是经校准的真实概率。
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("image 必须是 PIL.Image.Image，收到：{}".format(type(image)))
    if int(top_k) < 1:
        raise ValueError("top_k 必须大于等于 1")
    top_k = int(top_k)

    # 页面层应传入 RGB；这里兜底转换一次，避免 RGBA/L 模式导致仓库代码报错
    if image.mode != "RGB":
        image = image.convert("RGB")

    start = time.perf_counter()
    raw = predictor.predict(image, top_k=top_k)
    seconds = time.perf_counter() - start

    # 官方仓库返回 {label: score} 字典；此处做兼容处理并统一排序
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, (list, tuple)):
        items = [(item[0], item[1]) for item in raw]
    else:
        raise RuntimeError("模型返回了无法解析的结果类型：{}".format(type(raw)))

    predictions = [
        {"class_name": str(label), "score": float(score)} for label, score in items
    ]
    predictions.sort(key=lambda item: item["score"], reverse=True)
    return predictions, seconds
