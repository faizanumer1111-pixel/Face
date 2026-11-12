// Drives the whole web UI in a real browser and saves screenshots to tests/artifacts/.
//
//   ./run.sh --no-browser &              # start the app
//   npm install playwright && npx playwright install chromium   (once)
//   node tests/e2e/ui_flow.cjs [video] [face] [second face]
//
// Uses samples/test_video.mp4, samples/test_face.jpg and samples/test_face2.jpg by default.
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.join(ROOT, "tests", "artifacts");
const VIDEO = process.argv[2] || path.join(ROOT, "samples", "test_video.mp4");
const FACE = process.argv[3] || path.join(ROOT, "samples", "test_face.jpg");
const FACE2 = process.argv[4] || path.join(ROOT, "samples", "test_face2.jpg");
const URL = "http://127.0.0.1:7860";

async function shot(page, name, full = false) {
  await page.screenshot({ path: path.join(OUT, name), fullPage: full });
  console.log("screenshot", name);
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1, acceptDownloads: true });
  page.on("pageerror", (e) => { console.error("PAGE ERROR", e.message); process.exitCode = 1; });
  await page.goto(URL);
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.waitForFunction(() => document.getElementById("engine-text").textContent.startsWith("Ready"), null, { timeout: 180000 });
  await shot(page, "01-empty.png");

  // Faces first: two photos on the casting board
  const before = await page.locator("#talent .headshot").count();
  await page.setInputFiles("#file-face", [FACE, FACE2].filter((f) => fs.existsSync(f)));
  await page.waitForFunction((n) => document.querySelectorAll("#talent .headshot").length >= n + 1, before, { timeout: 120000 });

  await page.setInputFiles("#file-video", VIDEO);
  await page.waitForSelector("#video-wrap:not([hidden])", { timeout: 300000 });
  await page.waitForFunction(() => document.getElementById("video").readyState >= 2);
  await page.waitForSelector("#busy[hidden]", { state: "attached" });
  await shot(page, "02-loaded.png");

  const fps = await page.evaluate(() => S.fps);
  const n = Math.round(0.4 * fps);
  const mark = async (start, end, mode) => {
    await page.selectOption("#default-mode", mode);
    await page.evaluate((f) => seekFrame(f), start);
    await page.waitForTimeout(250);
    await page.locator("body").press("i");
    await page.evaluate((f) => seekFrame(f), end - 1);
    await page.waitForTimeout(250);
    await page.locator("body").press("o");
    await page.waitForTimeout(600);
  };
  await mark(25, 25 + n, "everyone");
  await mark(60, 60 + n, "cast");
  const rows = page.locator("#segments .tape-row");
  if ((await rows.count()) !== 2) throw new Error("expected 2 segments");
  console.log("savings:", await page.textContent("#savings"));

  // Casting board: drag a headshot onto each person in segment 2
  await page.evaluate((f) => seekFrame(f), 64);
  await page.waitForSelector(".vbox", { timeout: 120000 });
  const boxes = page.locator(".vbox");
  const count = await boxes.count();
  const sizes = [];
  for (let i = 0; i < count; i++) { const b = await boxes.nth(i).boundingBox(); sizes.push([b.width * b.height, i]); }
  sizes.sort((a, b) => b[0] - a[0]);
  const shots = page.locator("#talent .headshot");
  await shots.nth(0).dragTo(boxes.nth(sizes[0][1]));
  await page.waitForFunction(() => document.querySelectorAll("#segments .tape-row:nth-child(2) .cast-chip").length === 1, null, { timeout: 120000 });
  await page.waitForSelector(".vbox .tag", { timeout: 60000 });
  if (sizes.length > 1 && (await shots.count()) > 1) {
    await shots.nth(1).dragTo(page.locator(".vbox").nth(sizes[1][1]));
    await page.waitForFunction(() => document.querySelectorAll("#segments .tape-row:nth-child(2) .cast-chip").length === 2, null, { timeout: 120000 });
    await page.waitForFunction(() => document.querySelectorAll(".vbox .tag").length >= 2, null, { timeout: 60000 });
  }
  await page.waitForTimeout(500);
  await shot(page, "03-cast.png");

  // Click-to-cast works too (keyboard / no drag)
  await page.locator(".vbox").first().click();
  await page.waitForSelector("#picker:not([hidden]) button");
  await shot(page, "04-picker.png");
  await page.keyboard.press("Escape");

  // Preview segment 2
  await rows.nth(1).locator("button[data-act=preview]").click();
  await page.waitForSelector("#preview-dialog[open]", { timeout: 300000 });
  await page.waitForTimeout(500);
  await page.evaluate(() => { const s = document.getElementById("cmp-slider"); s.value = 0; s.dispatchEvent(new Event("input")); });
  console.log("preview:", await page.textContent("#preview-info"));
  await shot(page, "05-preview.png");
  await page.click("#preview-close");

  // Render
  await page.selectOption("#det-size", "320");
  await page.check("input[name=output][value=full]");
  await page.click("#btn-render");
  await page.waitForFunction(() => /Swapping|Saving|Done/.test(document.getElementById("progress-text").textContent), null, { timeout: 60000 });
  await page.locator("#render-panel").scrollIntoViewIfNeeded();
  await shot(page, "06-rendering.png");
  await page.waitForFunction(() => /^Done|failed|cancelled/.test(document.getElementById("progress-text").textContent), null, { timeout: 1500000 });
  const result = await page.textContent("#progress-text");
  console.log("render:", result);
  if (!result.startsWith("Done")) throw new Error("render didn't finish: " + result);
  await page.waitForSelector("#outputs .out-card");
  await shot(page, "07-done.png", true);

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#outputs .out-card a.button")]);
  const saved = path.join(OUT, "downloaded.mp4");
  await download.saveAs(saved);
  console.log("downloaded", download.suggestedFilename(), fs.statSync(saved).size, "bytes");

  // Narrow window: nothing should scroll sideways
  await page.setViewportSize({ width: 390, height: 900 });
  await page.waitForTimeout(300);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  console.log("phone-width horizontal overflow:", overflow);
  await shot(page, "08-narrow.png");
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
