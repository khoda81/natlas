// Run against tests/quantile_fixture.py; requires Playwright and Chromium.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH,
    args: ["--no-sandbox"],
  });
  const page = await browser.newPage({ viewport: { width: 1200, height: 850 } });
  const errors = [];
  const pending = new Set();
  let last = Date.now();
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (r.url().endsWith("/Quantile")) {
      pending.add(r);
      last = Date.now();
    }
  });
  for (const event of ["requestfinished", "requestfailed"])
    page.on(event, (r) => {
      pending.delete(r);
    });
  async function idle() {
    for (let count = 0; count < 400; count++) {
      await page.waitForTimeout(100);
      if (!pending.size && Date.now() - last > 700) return;
    }
    throw Error("Refinement did not settle");
  }
  await page.goto("http://127.0.0.1:8423/?server=http://127.0.0.1:8423");
  await page.waitForFunction(() => document.querySelector("#dataset").options.length === 2);
  await idle();
  assert.equal(await page.locator("#status").textContent(), "");
  assert.match(await page.locator("#readout").textContent(), /depth/);
  await page.screenshot({ path: "/tmp/natlas-main-v2-root.png" });
  await page.mouse.dblclick(100, 150);
  await page.waitForTimeout(500);
  await idle();
  assert.ok(!(await page.locator("#readout").textContent()).includes("NaN"));
  await page.keyboard.press("0");
  await page.locator("#q").selectOption("1");
  await idle();
  assert.equal(await page.locator("#status").textContent(), "");
  await page.locator("#swap").click();
  await idle();
  assert.equal(await page.locator("#status").textContent(), "");
  await page.keyboard.press("0");
  await page.mouse.move(400, 350);
  await page.mouse.wheel(0, -250);
  await idle();
  const afterDepth = await page.locator("#readout").textContent();
  assert.ok(!afterDepth.includes("NaN"));
  await page.keyboard.down("Control");
  await page.mouse.wheel(0, -100);
  await page.keyboard.up("Control");
  await idle();
  assert.notEqual(await page.locator("#readout").textContent(), afterDepth);
  await page.keyboard.press("0");
  await page.locator("#dataset").selectOption("qwen-prompt");
  await idle();
  assert.equal(await page.locator("#status").textContent(), "");
  assert.ok(!(await page.locator("#readout").textContent()).includes("NaN"));
  await page.screenshot({ path: "/tmp/natlas-main-v2-prompt.png" });
  await page.keyboard.press("0");
  await page.mouse.move(150, 200);
  await page.mouse.down();
  await page.mouse.move(200, 250, { steps: 5 });
  await page.mouse.up();
  await idle();
  assert.deepEqual(errors, []);
  assert.equal(await page.locator("#status").textContent(), "");
  console.log("Main camera: root, q switch, swap, depth, horizontal scale, prompt path and drag passed.");
  console.log(await (await page.request.get("http://127.0.0.1:8423/_test/state")).json());
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
