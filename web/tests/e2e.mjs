// End-to-end smoke test of the built single-file app: walks the whole lab loop in headless Chromium.
// Usage: npm run build && node tests/e2e.mjs [path/to/index.html] [screenshot-dir]
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

// CommonJS resolution so a globally installed Playwright (NODE_PATH) also works.
const { chromium } = createRequire(import.meta.url)("playwright");
const page_ = resolve(process.argv[2] ?? "dist/index.html");
const shots = process.argv[3];
if (shots) mkdirSync(shots, { recursive: true });
// E2E_PROXY=1 routes the browser through $HTTPS_PROXY (sandboxes where CDN scripts need a proxy).
const browser = await chromium.launch(process.env.E2E_PROXY ? { proxy: { server: process.env.HTTPS_PROXY } } : {});
const errors = [];
const run = async (colorScheme, width) => {
  const ctx = await browser.newContext({ viewport: { width, height: 900 }, colorScheme, ignoreHTTPSErrors: !!process.env.E2E_PROXY });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(`${colorScheme}/${width}: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && !m.text().includes("fonts.g") && errors.push(`${colorScheme}/${width}: ${m.text()}`));
  const shot = async (name) => shots && page.screenshot({ path: `${shots}/${colorScheme}-${width}-${name}.png`, fullPage: true });
  const step = async (hash) => {
    await page.click(`nav.tabs a[href="#${hash}"]`);
    await page.waitForSelector(".view-head h2");
  };
  const assertNoOverflow = async (name) => {
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (over > 1) errors.push(`${colorScheme}/${width}/${name}: page scrolls horizontally by ${over}px`);
  };
  await page.goto(pathToFileURL(page_).href);
  await page.waitForSelector(".batch-strip .batch");
  await assertNoOverflow("bench");
  await shot("1-bench");

  await step("campaign");
  await page.click("button:has-text('Run campaign')");
  await page.waitForSelector("table tbody tr");
  await assertNoOverflow("campaign");
  await shot("2-campaign");

  await step("model");
  await page.click("button:has-text('Train model')");
  await page.waitForSelector("text=What the model reacts to", { timeout: 120_000 });
  await assertNoOverflow("model");
  await shot("3-model");

  await step("optimise");
  await page.selectOption("#budget", "800");
  await page.selectOption("#deploy-n", "300");
  await page.click("button:has-text('Optimise and deploy')");
  await page.waitForSelector("text=Recommended settings and what happened in the simulator", { timeout: 180_000 });
  const rows = await page.locator("table tbody tr").count();
  if (rows !== 6) errors.push(`${colorScheme}/${width}: expected 6 recommendations (3 objectives × 2 models), got ${rows}`);
  await assertNoOverflow("optimise");
  await shot("4-optimise");

  await step("experiments");
  await page.click("button:has-text('Propose experiments')");
  await page.waitForSelector("text=Proposed jobs", { timeout: 120_000 });
  await page.click("button:has-text('Print them and retrain')");
  await page.waitForFunction(() => !document.querySelector(".progress"), null, { timeout: 120_000 });
  await page.click("button:has-text('Propose experiments')");
  await page.waitForSelector("text=Is it learning?", { timeout: 120_000 });
  await assertNoOverflow("experiments");
  await shot("5-experiments");

  await step("study");
  await page.waitForSelector("text=E1: robust vs nominal");
  await assertNoOverflow("study");
  await shot("6-study");
  await ctx.close();
};
// Imported data: round-trip the campaign CSV through the import panel; Optimise must then
// report model predictions only (the simulator's truth says nothing about a real material).
const runImported = async () => {
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 900 }, ignoreHTTPSErrors: !!process.env.E2E_PROXY });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(`import: ${e.message}`));
  await page.goto(pathToFileURL(page_).href + "#campaign");
  await page.click("button:has-text('Run campaign')");
  await page.click("button:has-text('Show CSV')");
  const csv = await page.inputValue("#csv-out");
  await page.fill("#csv-paste", csv);
  await page.click("button:has-text('Validate')");
  await page.click("button:has-text('valid jobs')");
  await page.click('nav.tabs a[href="#model"]');
  await page.click("button:has-text('Train model')");
  await page.waitForSelector("text=What the model reacts to", { timeout: 120_000 });
  await page.click('nav.tabs a[href="#optimise"]');
  await page.selectOption("#budget", "800");
  await page.click("button:has-text('Optimise and deploy')");
  await page.waitForSelector("h3:has-text('Recommended settings')", { timeout: 180_000 });
  if (await page.locator("text=Failed prints on fresh batches").count()) errors.push("import: simulated deployment shown for imported data");
  if (!(await page.locator("th:has-text('Pred. CVaR 10 %')").count())) errors.push("import: predicted columns missing");
  await ctx.close();
};
const t0 = Date.now();
await run("light", 1360);
await run("dark", 390);
await runImported();
await browser.close();
console.log(`e2e finished in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log("e2e: no errors");
