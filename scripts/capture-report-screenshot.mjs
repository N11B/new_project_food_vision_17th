import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const imagePath = path.join(root, "test_images", "pizza_2582289.jpg");
const outputPath = path.join(root, "screenshots", "foodvision_nutrition_result.png");
const debugBase = "http://127.0.0.1:9223";
const pageUrl = `http://127.0.0.1:4174/?screenshot=${Date.now()}`;

const targetResponse = await fetch(
  `${debugBase}/json/new?${encodeURIComponent(pageUrl)}`,
  { method: "PUT" },
);
if (!targetResponse.ok) throw new Error(`无法建立截图页面：${targetResponse.status}`);
const target = await targetResponse.json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
const pending = new Map();
let sequence = 0;

await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (!message.id || !pending.has(message.id)) return;
  const { resolve, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message));
  else resolve(message.result);
});

function send(method, params = {}) {
  sequence += 1;
  const id = sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}

async function waitFor(expression, timeoutMs = 120000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await evaluate(expression)) return;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`等待页面状态超时：${expression}`);
}

await send("Page.enable");
await send("Runtime.enable");
await send("DOM.enable");
await send("Network.enable");
await send("Network.setCacheDisabled", { cacheDisabled: true });
await send("Emulation.setDeviceMetricsOverride", {
  width: 1440,
  height: 1180,
  deviceScaleFactor: 1,
  mobile: false,
});
await send("Page.navigate", { url: pageUrl });
await waitFor("document.readyState === 'complete'", 30000);

const documentNode = await send("DOM.getDocument", { depth: -1, pierce: true });
const inputNode = await send("DOM.querySelector", {
  nodeId: documentNode.root.nodeId,
  selector: "#fileInput",
});
if (!inputNode.nodeId) throw new Error("没有找到图片上传控件");
await send("DOM.setFileInputFiles", { nodeId: inputNode.nodeId, files: [imagePath] });
await waitFor("document.querySelector('#predictButton')?.disabled === false", 30000);
await evaluate("document.querySelector('#predictButton').click(); true");
await waitFor("document.querySelector('#resultContent')?.hidden === false", 120000);
await waitFor("document.querySelector('#nutritionTypical')?.textContent !== '—'", 30000);

const resultText = await evaluate(`({
  food: document.querySelector('#topClass').textContent,
  score: document.querySelector('#topScore').textContent,
  kcal: document.querySelector('#nutritionTypical').textContent,
  range: document.querySelector('#nutritionRange').textContent
})`);
const layout = await send("Page.getLayoutMetrics");
const screenshot = await send("Page.captureScreenshot", {
  format: "png",
  captureBeyondViewport: true,
  clip: {
    x: 0,
    y: 0,
    width: Math.min(Math.ceil(layout.cssContentSize.width), 1440),
    height: Math.ceil(layout.cssContentSize.height),
    scale: 1,
  },
});

await writeFile(outputPath, Buffer.from(screenshot.data, "base64"));
socket.close();
console.log(JSON.stringify({ outputPath, ...resultText }, null, 2));
