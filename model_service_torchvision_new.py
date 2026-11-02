# -*- coding: utf-8 -*-
"""model_service_torchvision_new.py —— torchvision >= 0.13 专用加载后端（推荐）。

适用环境：torchvision 0.13 及以上（此时 torchvision 提供 Weights 枚举 API，
官方 model.py 可以原样加载，无需任何兼容处理）。

接口与 model_service_torchvision_old.py 完全相同：
    predictor, device_name = load_predictor()

通常不直接使用本文件；统一入口 model_service.py 会根据 torchvision 版本
自动选择本文件或 model_service_torchvision_old.py。
"""


def load_predictor():
    """加载 FoodVision Big 模型，返回 (predictor, device_name)。

    - 模型文件来自本地 foodvision_big/ 目录（缺失时自动下载）；
    - 加载后做一次预热推理，CUDA 失败时自动回退 CPU。
    """
    # 延迟导入，避免与统一入口 model_service.py 循环依赖
    from model_service import (
        ensure_model_downloaded,
        import_repo_model,
        load_repo_predictor_with_fallback,
    )

    model_dir = ensure_model_downloaded()
    repo_model = import_repo_model(model_dir)
    return load_repo_predictor_with_fallback(repo_model, model_dir)


if __name__ == "__main__":
    predictor, device = load_predictor()
    print("torchvision>=0.13 后端自检通过，设备:", device)
