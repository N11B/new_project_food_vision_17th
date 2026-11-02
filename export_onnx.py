"""将本地 FoodVision Big PyTorch 权重导出为浏览器可用的 ONNX 模型。

本脚本只用于开发/重新导出，不会在 Vercel 构建时运行。网页运行时只需要
public/model/foodvision-big.onnx 和 public/model/classes.json。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
PUBLIC_MODEL_DIR = ROOT / "public" / "model"
ONNX_PATH = PUBLIC_MODEL_DIR / "foodvision-big.onnx"
CLASSES_PATH = PUBLIC_MODEL_DIR / "classes.json"
TEST_IMAGE = ROOT / "test_images" / "pizza_2582289.jpg"


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def main() -> None:
    os.environ["FOODVISION_DEVICE"] = "cpu"
    sys.path.insert(0, str(ROOT))

    from model_service import load_predictor

    PUBLIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor, device = load_predictor()
    model = predictor.model.eval().cpu()

    dummy = torch.zeros(1, 3, 288, 288, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
        do_constant_folding=True,
    )

    model_proto = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model_proto)
    CLASSES_PATH.write_text(
        json.dumps(predictor.class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 用原项目测试图片验证 PyTorch 与 ONNX Runtime 输出一致。
    with Image.open(TEST_IMAGE) as raw_image:
        image = raw_image.convert("RGB")
    tensor = predictor.transform(image).unsqueeze(0)
    with torch.inference_mode():
        torch_probs = torch.softmax(model(tensor), dim=1).cpu().numpy()

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"image": tensor.numpy()})[0]
    onnx_probs = softmax(logits)

    max_abs_diff = float(np.max(np.abs(torch_probs - onnx_probs)))
    torch_top1 = int(torch_probs.argmax(axis=1)[0])
    onnx_top1 = int(onnx_probs.argmax(axis=1)[0])
    if torch_top1 != onnx_top1 or max_abs_diff > 1e-4:
        raise RuntimeError(
            f"ONNX 验证失败: torch_top1={torch_top1}, onnx_top1={onnx_top1}, "
            f"max_abs_diff={max_abs_diff}"
        )

    print(f"导出完成: {ONNX_PATH}")
    print(f"模型大小: {ONNX_PATH.stat().st_size / 1024 / 1024:.2f} MiB")
    print(f"验证设备: {device}")
    print(f"Top-1: {predictor.class_names[onnx_top1]} ({onnx_probs[0, onnx_top1]:.6f})")
    print(f"PyTorch / ONNX 最大概率差: {max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
