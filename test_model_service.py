# -*- coding: utf-8 -*-
"""A 方向自测脚本（成员 A 使用，不属于 app 集成代码）。

测试内容对应《02_详细代码规划方案》第 6 节的 1~3 步：
1. 检查 torch / torchvision / CUDA 环境；
2. 调用 load_predictor()，确认模型成功加载；
3. 用真实图片调用 predict_topk()，确认返回非空、降序、格式正确的预测列表；
4. 复测一次加载缓存行为（重复调用 load_predictor 不重复下载文件）。

运行方式（在 tft-torch 环境中，于"代码"目录下执行）：
    python test_model_service.py
    python test_model_service.py path/to/your.jpg   # 指定自己的测试图片
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model_service  # noqa: E402


def step1_check_env():
    import torch
    import torchvision

    print("[1] 环境检查")
    print("    python       :", sys.version.split()[0])
    print("    torch        :", torch.__version__)
    print("    torchvision  :", torchvision.__version__)
    print("    cuda 可用    :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("    GPU          :", torch.cuda.get_device_name(0))


def get_test_image():
    """获取一张真实测试图片：优先用命令行参数，其次本地 test_images，最后官方示例图/本地生成图兜底。"""
    from PIL import Image

    # 1) 命令行指定的图片
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        image = Image.open(path).convert("RGB")
        print("    使用指定图片 :", path)
        return image

    # 2) 本地 test_images 目录下的图片（随代码保存，离线可用）
    local_dir = HERE / "test_images"
    local_images = sorted(local_dir.glob("*.jpg")) + sorted(local_dir.glob("*.png")) if local_dir.is_dir() else []
    if local_images:
        image = Image.open(local_images[0]).convert("RGB")
        print("    使用本地测试图:", local_images[0])
        return image

    # 3) 官方示例图（Space 仓库里的 examples/2582289.jpg，一张披萨照片）
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            model_service.MODEL_REPO_ID, "examples/2582289.jpg", repo_type="space"
        )
        image = Image.open(local).convert("RGB")
        print("    使用官方示例图:", local)
        return image
    except Exception as exc:
        print("    官方示例图获取失败（不影响主流程）：", exc)

    # 4) 兜底：本地生成一张纯色图（仅验证接口连通性，预测结果无意义）
    print("    使用本地生成图（兜底，仅验证流程）")
    buf = io.BytesIO()
    Image.new("RGB", (320, 320), (200, 120, 60)).save(buf, format="JPEG")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    step1_check_env()

    print("[2] 加载模型")
    predictor, device = model_service.load_predictor()
    print("    加载成功，设备:", device, "后端:", model_service.BACKEND)

    image = get_test_image()

    print("[3] Top-5 预测")
    predictions, seconds = model_service.predict_topk(predictor, image, top_k=5)
    print("    耗时: {:.4f} 秒".format(seconds))
    for i, item in enumerate(predictions, 1):
        print("    Top-{}: {:<28s} score={:.4f}".format(i, item["class_name"], item["score"]))

    print("[4] 结果校验")
    assert len(predictions) == 5, "Top-5 应返回 5 条预测"
    scores = [item["score"] for item in predictions]
    assert scores == sorted(scores, reverse=True), "分数应为降序"
    assert all(set(item) == {"class_name", "score"} for item in predictions), "字段格式不符"
    assert isinstance(seconds, float) and seconds >= 0, "耗时格式不符"
    print("    全部通过")

    print("[5] 复测：再次加载与预测（验证模型文件已本地就绪、接口稳定）")
    predictor2, device2 = model_service.load_predictor()
    predictions2, _ = model_service.predict_topk(predictor2, image, top_k=1)
    print("    Top-1:", predictions2[0]["class_name"], "设备:", device2)
    print("\nA 方向自测全部通过。")


if __name__ == "__main__":
    main()
