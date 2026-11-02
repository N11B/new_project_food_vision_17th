import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const classes = JSON.parse(await readFile(path.join(root, "public/model/classes.json"), "utf8"));
const database = JSON.parse(await readFile(path.join(root, "public/data/nutrition.json"), "utf8"));
const foods = database.foods ?? {};

const classSet = new Set(classes);
const foodKeys = Object.keys(foods);
const missing = classes.filter((className) => !foods[className]);
const extra = foodKeys.filter((className) => !classSet.has(className));
const invalid = foodKeys.filter((className) => {
  const item = foods[className];
  return (
    typeof item.name_zh !== "string" ||
    !item.name_zh.trim() ||
    !Number.isFinite(item.kcal_per_100g) ||
    item.kcal_per_100g <= 0 ||
    !Number.isFinite(item.variability) ||
    item.variability <= 0 ||
    item.variability >= 1
  );
});

if (new Set(classes).size !== classes.length) throw new Error("classes.json 存在重复类别");
if (missing.length) throw new Error(`营养数据库缺少类别：${missing.join(", ")}`);
if (extra.length) throw new Error(`营养数据库存在多余类别：${extra.join(", ")}`);
if (invalid.length) throw new Error(`营养数据字段无效：${invalid.join(", ")}`);

console.log(`营养数据库校验通过：${classes.length} 个 Food-101 类别完整且字段有效。`);
