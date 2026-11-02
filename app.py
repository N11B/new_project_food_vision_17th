# -*- coding: utf-8 -*-
"""app.py —— 成员 B 负责：Streamlit 网页界面、输入交互与结果展示。

通过 model_service.py 的冻结接口调用模型，本文件不包含任何模型加载细节。
"""

import io
import streamlit as st
from PIL import Image
from model_service import load_predictor, predict_topk

st.set_page_config(page_title="食鉴 AI", layout="centered")

st.title("食鉴 AI：基于图像的食品分类")

st.caption(
    "本应用使用 FoodVision Big 预训练模型（基于 EfficientNet-B2），"
    "仅覆盖 Food-101 的 101 类菜品。范围外的食物或非食品图片也可能被归到某一类，"
    "模型输出分数不是经校准的真实概率。"
)


@st.cache_resource
def get_model():
    return load_predictor()


input_mode = st.radio(
    "选择输入方式",
    ["上传图片", "摄像头拍照"],
    horizontal=True,
)

if input_mode == "上传图片":
    source = st.file_uploader(
        "选择一张食品图片",
        type=["jpg", "jpeg", "png"],
    )
else:
    source = st.camera_input("拍摄一张食品照片")

if source is not None:
    try:
        image = Image.open(io.BytesIO(source.getvalue())).convert("RGB")
    except Exception:
        st.error("图片无法解码，请换一张图片重试。")
        st.stop()

    st.image(image, caption="待识别图片", width='content')

    if st.button("识别食品", type="primary"):
        with st.spinner("正在识别，请稍候……"):
            try:
                predictor, device = get_model()
                predictions, seconds = predict_topk(predictor, image, top_k=5)
            except Exception as exc:
                st.error(f"识别失败：{exc}")
                st.stop()

        st.success(
            f"设备：{device}  耗时：{seconds:.4f} 秒"
        )

        top1 = predictions[0]
        st.write(
            f"**Top-1：{top1['class_name']}**（分数：{top1['score']:.4f}）"
        )

        st.write("**Top-5 预测结果：**")
        st.table(
            {
                "排名": [f"#{i+1}" for i in range(len(predictions))],
                "类别": [p["class_name"] for p in predictions],
                "分数": [f"{p['score']:.4f}" for p in predictions],
            }
        )
