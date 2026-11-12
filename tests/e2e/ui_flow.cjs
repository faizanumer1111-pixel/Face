// Drives the whole web UI in a real browser and saves screenshots to tests/artifacts/.
//
//   ./run.sh --no-browser &              # start the app
//   npm install playwright && npx playwright install chromium   (once)
//   node tests/e2e/ui_flow.cjs [video] [face]
//
// Uses samples/test_video.mp4 and samples/test_face.jpg by default.
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "tests", "artifacts");
const VIDEO = process.argv[2] || path.join(ROOT, "samples", "test_video.mp4");
const FACE = process.argv[3] || path.join(ROOT, "samples", "test_face.jpg");
const URL = "http://127.0.0.1:7860";

async function shot(page, name) {
  await page.screenshot({ path: path.join(OUT, name), fullPage: true });
  console.log("screenshot", name);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
  const page = await browser.newPage({ viewport: { width: 1360, height: 1000 }, acceptDownloads: true });
  page.on("pageerror", (e) => { console.error("PAGE ERROR", e.message); process.exitCode = 1; });
  await page.goto(URL);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.waitForFunction(() => document.getElementById("engine-status").textContent.startsWith("Ready"), null, { timeout: 120000 });
  await shot(page, "01-start.png");

  await page.setInputFiles("#file-video", VIDEO);
  await page.waitForSelector("#editor:not([hidden])", { timeout: 120000 });
  await page.waitForFunction(() => document.getElementById("video").readyState >= 2);
  await page.setInputFiles("#file-face", FACE);
  await page.waitForSelector("#face-pick:not([hidden])", { timeout: 60000 });
  await shot(page, "02-inputs.png");

  const fps = await page.evaluate(() => S.fps);
  const mark = async (start, end, mode) => {
    await page.selectOption("#default-mode", mode);
    await page.evaluate((f) => seekFrame(f), start);
    await page.waitForTimeout(250);
    await page.locator("body").press("i");
    await page.evaluate((f) => seekFrame(f), end - 1);
    await page.waitForTimeout(250);
    await page.locator("body").press("o");
    await page.waitForTimeout(500);
  };
  // Segment 1: everyone, ~0.4s. Segment 2: one person, ~0.4s.
  const n = Math.round(0.4 * fps);
  await mark(25, 25 + n, "everyone");
  await mark(60, 60 + n, "one");
  const rows = page.locator("#segments tr");
  if ((await rows.count()) !== 2) throw new Error("expected 2 segments");
  const savings = await page.textContent("#savings");
  console.log("savings:", savings);
  if (!savings.includes(`Processing ${2 * n} of`)) throw new Error("savings line is wrong: " + savings);

  // Pick the person for segment 2
  await rows.nth(1).locator("button[data-act=pick]").click();
  await page.waitForSelector(".vbox", { timeout: 60000 });
  await shot(page, "03-pick-person.png");
  const boxes = page.locator(".vbox");
  // pick the biggest box
  let best = 0, bestArea = 0;
  for (let i = 0; i < (await boxes.count()); i++) {
    const b = await boxes.nth(i).boundingBox();
    if (b.width * b.height > bestArea) { bestArea = b.width * b.height; best = i; }
  }
  await boxes.nth(best).click();
  await page.waitForSelector("#segments tr:nth-child(2) .who img", { timeout: 60000 });

  // Preview a frame of segment 2
  await rows.nth(1).locator("button[data-act=preview]").click();
  await page.waitForSelector("#preview-dialog[open]", { timeout: 120000 });
  await page.waitForTimeout(400);
  console.log("preview:", await page.textContent("#preview-info"));
  await shot(page, "04-preview.png");
  await page.click("#preview-close");
  await shot(page, "05-editor.png");

  // Render the full video with fast detection
  await page.check("input[name=output][value=full]");
  await page.selectOption("#det-size", "320");
  await page.click("#btn-render");
  await page.waitForFunction(() => /Swapping|Finishing|Done/.test(document.getElementById("progress-text").textContent), null, { timeout: 60000 });
  await shot(page, "06-rendering.png");
  await page.waitForFunction(() => /^Done|failed|cancelled/.test(document.getElementById("progress-text").textContent), null, { timeout: 900000 });
  const result = await page.textContent("#progress-text");
  console.log("render:", result);
  if (!result.startsWith("Done")) throw new Error("render didn't finish: " + result);
  await page.waitForSelector("#outputs li");
  await shot(page, "07-done.png");

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#outputs li a.button")]);
  const saved = path.join(OUT, "downloaded.mp4");
  await download.saveAs(saved);
  console.log("downloaded", download.suggestedFilename(), fs.statSync(saved).size, "bytes");
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
