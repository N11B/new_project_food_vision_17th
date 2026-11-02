# -*- coding: utf-8 -*-
"""model_service_torchvision_old.py —— torchvision < 0.13 专用加载后端（不推荐）。

适用环境：torchvision 0.11.x / 0.12.x（例如本机 torch 1.10.0+cu113 +
torchvision 0.11.1，Python 3.9）。该版本区间没有 Weights 枚举 API，
官方 model.py 无法直接加载，因此本文件提供等效兼容层：

- EfficientNet_B2_Weights.IMAGENET1K_V1 在旧版中的语义就是
  "resize 到 288（bicubic）-> center crop 288 -> ImageNet mean/std"，
  兼容层按官方模型卡给出完全一致的预处理；
- 推理时权重全部来自本地 model.safetensors，不触发任何 torchvision 下载。

接口与 model_service_torchvision_new.py 完全相同：
    predictor, device_name = load_predictor()

注意：交接环境若能安装 torchvision >= 0.13，请优先使用
model_service_torchvision_new.py（统一入口会自动选择）。
"""

import warnings


def _ensure_torchvision_compat():
    """torchvision < 0.13 时，为官方 model.py 注入等效的 Weights 兼容层。"""
    import torchvision.models as tvm

    if hasattr(tvm, "EfficientNet_B2_Weights"):
        return  # torchvision >= 0.13，无需处理

    from torchvision import transforms as T

    class _B2WeightsShim:
        """等效于 EfficientNet_B2_Weights.IMAGENET1K_V1 的最小实现。"""

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        def transforms(self):
            return T.Compose(
                [
                    T.Resize(288, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(288),
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std),
                ]
            )

        def __repr__(self):
            return "EfficientNet_B2_Weights.IMAGENET1K_V1 (compat shim)"

    _weights = _B2WeightsShim()

    class _B2WeightsEnumShim:
        IMAGENET1K_V1 = _weights
        DEFAULT = _weights

    tvm.EfficientNet_B2_Weights = _B2WeightsEnumShim

    # 旧版 efficientnet_b2 不认识 weights 关键字；推理时权重来自 safetensors，直接剥掉
    _orig_efficientnet_b2 = tvm.efficientnet_b2

    def _efficientnet_b2_compat(*args, **kwargs):
        kwargs.pop("weights", None)
        return _orig_efficientnet_b2(*args, **kwargs)

    tvm.efficientnet_b2 = _efficientnet_b2_compat
    warnings.warn(
        "检测到 torchvision < 0.13，已启用 EfficientNet_B2_Weights 兼容层"
        "（预处理与官方模型卡一致）",
        stacklevel=2,
    )


def load_predictor():
    """加载 FoodVision Big 模型，返回 (predictor, device_name)。

    - 先注入 torchvision 兼容层，再加载官方 model.py；
    - 模型文件来自本地 foodvision_big/ 目录（缺失时自动下载）；
    - 加载后做一次预热推理，CUDA 失败时自动回退 CPU。
    """
    # 延迟导入，避免与统一入口 model_service.py 循环依赖
    from model_service import (
        ensure_model_downloaded,
        import_repo_model,
        load_repo_predictor_with_fallback,
    )

    _ensure_torchvision_compat()
    model_dir = ensure_model_downloaded()
    repo_model = import_repo_model(model_dir)
    return load_repo_predictor_with_fallback(repo_model, model_dir)


if __name__ == "__main__":
    predictor, device = load_predictor()
    print("torchvision<0.13 后端自检通过，设备:", device)
