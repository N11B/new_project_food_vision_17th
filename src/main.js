import * as ort from "onnxruntime-web";
import "./style.css";

const MODEL_URL = "/model/foodvision-big.onnx";
const CLASSES_URL = "/model/classes.json";
const NUTRITION_URL = "/data/nutrition.json";
const IMAGE_SIZE = 288;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const elements = {
  dropZone: document.querySelector("#dropZone"),
  fileInput: document.querySelector("#fileInput"),
  cameraInput: document.querySelector("#cameraInput"),
  chooseButton: document.querySelector("#chooseButton"),
  cameraButton: document.querySelector("#cameraButton"),
  replaceButton: document.querySelector("#replaceButton"),
  emptyState: document.querySelector("#emptyState"),
  previewState: document.querySelector("#previewState"),
  previewImage: document.querySelector("#previewImage"),
  predictButton: document.querySelector("#predictButton"),
  predictButtonText: document.querySelector("#predictButtonText"),
  buttonSpinner: document.querySelector("#buttonSpinner"),
  modelStatus: document.querySelector("#modelStatus"),
  statusDot: document.querySelector("#statusDot"),
  errorMessage: document.querySelector("#errorMessage"),
  resultPlaceholder: document.querySelector("#resultPlaceholder"),
  resultContent: document.querySelector("#resultContent"),
  inferenceMeta: document.querySelector("#inferenceMeta"),
  topClass: document.querySelector("#topClass"),
  topClassEnglish: document.querySelector("#topClassEnglish"),
  topScore: document.querySelector("#topScore"),
  predictionList: document.querySelector("#predictionList"),
  nutritionTypical: document.querySelector("#nutritionTypical"),
  nutritionRange: document.querySelector("#nutritionRange"),
  portionInput: document.querySelector("#portionInput"),
  portionEstimate: document.querySelector("#portionEstimate"),
};

let selectedImage = null;
let selectedObjectUrl = null;
let modelPromise = null;
let currentNutrition = null;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

function setStatus(message, type = "loading") {
  elements.modelStatus.textContent = message;
  elements.statusDot.className = `status-dot ${type}`;
}

function setError(message = "") {
  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = !message;
}

async function getModel() {
  if (!modelPromise) {
    setStatus("正在下载并初始化模型…", "loading");
    modelPromise = Promise.all([
      ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
      fetch(CLASSES_URL).then((response) => {
        if (!response.ok) throw new Error("类别配置加载失败");
        return response.json();
      }),
      fetch(NUTRITION_URL).then((response) => {
        if (!response.ok) throw new Error("营养数据库加载失败");
        return response.json();
      }),
    ])
      .then(([session, classes, nutrition]) => {
        setStatus("模型已就绪 · 浏览器本地推理", "ready");
        return { session, classes, nutrition };
      })
      .catch((error) => {
        modelPromise = null;
        setStatus("模型加载失败", "error");
        throw error;
      });
  }
  return modelPromise;
}

function isSupportedImage(file) {
  return file && ["image/jpeg", "image/png"].includes(file.type);
}

function loadImageFromUrl(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("图片无法读取，请换一张图片重试"));
    image.src = url;
  });
}

async function selectFile(file) {
  setError();
  if (!isSupportedImage(file)) {
    setError("仅支持 JPG、JPEG 或 PNG 图片");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    setError("图片不能超过 20 MB");
    return;
  }

  try {
    if (selectedObjectUrl) URL.revokeObjectURL(selectedObjectUrl);
    selectedObjectUrl = URL.createObjectURL(file);
    selectedImage = await loadImageFromUrl(selectedObjectUrl);
    elements.previewImage.src = selectedObjectUrl;
    elements.emptyState.hidden = true;
    elements.previewState.hidden = false;
    elements.predictButton.disabled = false;
    elements.resultContent.hidden = true;
    elements.resultPlaceholder.hidden = false;
    elements.inferenceMeta.textContent = "等待识别";
  } catch (error) {
    setError(error.message);
  }
}

function preprocess(image) {
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  // torchvision Resize(288) + CenterCrop(288) 等价于先取源图中央正方形再缩放。
  const cropSize = Math.min(image.naturalWidth, image.naturalHeight);
  const sourceX = (image.naturalWidth - cropSize) / 2;
  const sourceY = (image.naturalHeight - cropSize) / 2;
  context.drawImage(
    image,
    sourceX,
    sourceY,
    cropSize,
    cropSize,
    0,
    0,
    IMAGE_SIZE,
    IMAGE_SIZE,
  );

  const rgba = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const channelSize = IMAGE_SIZE * IMAGE_SIZE;
  const tensorData = new Float32Array(3 * channelSize);

  for (let pixel = 0; pixel < channelSize; pixel += 1) {
    const rgbaIndex = pixel * 4;
    tensorData[pixel] = (rgba[rgbaIndex] / 255 - MEAN[0]) / STD[0];
    tensorData[channelSize + pixel] = (rgba[rgbaIndex + 1] / 255 - MEAN[1]) / STD[1];
    tensorData[channelSize * 2 + pixel] = (rgba[rgbaIndex + 2] / 255 - MEAN[2]) / STD[2];
  }

  return new ort.Tensor("float32", tensorData, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function softmax(logits) {
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  const probabilities = new Float32Array(logits.length);
  let sum = 0;
  for (let index = 0; index < logits.length; index += 1) {
    probabilities[index] = Math.exp(logits[index] - max);
    sum += probabilities[index];
  }
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] /= sum;
  }
  return probabilities;
}

function displayName(className) {
  return className
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function calorieRange(record, weight = 100) {
  const scale = weight / 100;
  const typical = Math.round(record.kcal_per_100g * scale);
  const low = Math.round(record.kcal_per_100g * (1 - record.variability) * scale);
  const high = Math.round(record.kcal_per_100g * (1 + record.variability) * scale);
  return { typical, low, high };
}

function updatePortionEstimate() {
  const weight = Number(elements.portionInput.value);
  if (!currentNutrition || !Number.isFinite(weight) || weight <= 0 || weight > 2000) {
    elements.portionEstimate.textContent = "请输入 1–2000 g";
    return;
  }
  const { typical, low, high } = calorieRange(currentNutrition, weight);
  elements.portionEstimate.textContent = `约 ${typical} kcal（${low}–${high} kcal）`;
}

function renderNutrition(record) {
  currentNutrition = record;
  const { typical, low, high } = calorieRange(record);
  elements.nutritionTypical.textContent = typical;
  elements.nutritionRange.textContent = `参考区间 ${low}–${high} kcal`;
  updatePortionEstimate();
}

function renderResults(predictions, elapsedMs, nutrition) {
  const top = predictions[0];
  const nutritionRecord = nutrition.foods[top.className];
  if (!nutritionRecord) throw new Error(`营养数据库缺少类别：${top.className}`);

  elements.topClass.textContent = nutritionRecord.name_zh;
  elements.topClassEnglish.textContent = displayName(top.className);
  elements.topScore.textContent = `${(top.score * 100).toFixed(2)}%`;
  elements.inferenceMeta.textContent = `WebAssembly · ${elapsedMs.toFixed(0)} ms`;
  elements.predictionList.replaceChildren();
  renderNutrition(nutritionRecord);

  predictions.forEach((prediction, index) => {
    const record = nutrition.foods[prediction.className];
    const item = document.createElement("li");
    item.innerHTML = `
      <span class="rank">${String(index + 1).padStart(2, "0")}</span>
      <div class="prediction-main">
        <div class="prediction-label">
          <span>${record ? `${record.name_zh} · ` : ""}${displayName(prediction.className)}</span>
          <strong>${(prediction.score * 100).toFixed(2)}%</strong>
        </div>
        <div class="score-track"><span style="width: ${Math.max(prediction.score * 100, 1)}%"></span></div>
      </div>
    `;
    elements.predictionList.append(item);
  });

  elements.resultPlaceholder.hidden = true;
  elements.resultContent.hidden = false;
}

async function predict() {
  if (!selectedImage) return;
  setError();
  elements.predictButton.disabled = true;
  elements.predictButtonText.textContent = "正在识别";
  elements.buttonSpinner.hidden = false;

  try {
    const { session, classes, nutrition } = await getModel();
    const input = preprocess(selectedImage);
    const startedAt = performance.now();
    const outputs = await session.run({ [session.inputNames[0]]: input });
    const elapsedMs = performance.now() - startedAt;
    const logits = outputs[session.outputNames[0]].data;
    const probabilities = softmax(logits);
    const predictions = Array.from(probabilities, (score, index) => ({
      className: classes[index],
      score,
    }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
    renderResults(predictions, elapsedMs, nutrition);
  } catch (error) {
    console.error(error);
    setError(`识别失败：${error.message || "请刷新页面后重试"}`);
  } finally {
    elements.predictButton.disabled = false;
    elements.predictButtonText.textContent = "重新识别";
    elements.buttonSpinner.hidden = true;
  }
}

elements.chooseButton.addEventListener("click", (event) => {
  event.stopPropagation();
  elements.fileInput.click();
});
elements.cameraButton.addEventListener("click", (event) => {
  event.stopPropagation();
  elements.cameraInput.click();
});
elements.replaceButton.addEventListener("click", (event) => {
  event.stopPropagation();
  elements.fileInput.click();
});
elements.fileInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
elements.cameraInput.addEventListener("change", (event) => selectFile(event.target.files[0]));
elements.predictButton.addEventListener("click", predict);
elements.portionInput.addEventListener("input", updatePortionEstimate);

elements.dropZone.addEventListener("click", () => {
  if (!selectedImage) elements.fileInput.click();
});
elements.dropZone.addEventListener("keydown", (event) => {
  if ((event.key === "Enter" || event.key === " ") && !selectedImage) {
    event.preventDefault();
    elements.fileInput.click();
  }
});
for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}
elements.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

// 空闲时提前下载模型；不会阻塞页面首次渲染。
if ("requestIdleCallback" in window) {
  window.requestIdleCallback(() => getModel().catch(() => {}), { timeout: 2500 });
}
