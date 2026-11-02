# -*- coding: utf-8 -*-
"""自动截取 Streamlit 页面截图"""
import time
from playwright.sync_api import sync_playwright

SCREENSHOT_DIR = r"G:\南航\食鉴AI_统一极简方案包\代码\screenshots"
BASE_URL = "http://localhost:8501"
PIZZA_IMAGE = r"G:\南航\食鉴AI_统一极简方案包\代码\test_images\pizza_2582289.jpg"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    # 1. 首页截图
    page.goto(BASE_URL)
    page.wait_for_timeout(2000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/01_homepage.png", full_page=True)
    print("截图 1: 首页界面 ✓")

    # 2. 上传 pizza 图
    file_input = page.locator('input[type="file"]')
    file_input.set_input_files(PIZZA_IMAGE)
    page.wait_for_timeout(2000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/02_image_preview.png", full_page=True)
    print("截图 2: 图片预览 ✓")

    # 3. 点击识别，等待结果
    page.get_by_role("button", name="识别食品").click()
    page.wait_for_timeout(5000)  # 等待模型推理
    page.screenshot(path=f"{SCREENSHOT_DIR}/03_result.png", full_page=True)
    print("截图 3: 识别结果 ✓")

    # 4. 切换到摄像头模式
    page.get_by_role("radio", name="摄像头拍照").click()
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SCREENSHOT_DIR}/04_camera_mode.png", full_page=True)
    print("截图 4: 摄像头模式 ✓")

    browser.close()
    print("\n所有截图已保存到:", SCREENSHOT_DIR)
