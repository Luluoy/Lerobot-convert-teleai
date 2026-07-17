import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { rm } from "node:fs/promises";

const baseURL = process.env.UI_BASE_URL || "http://127.0.0.1:8765";
const python = process.env.UI_PYTHON || process.env.LEROBOT_DATACONVERT_PYTHON || "python";
const outputPath = "/tmp/lerobot-dataconvert-ui-e2e";
const cachePath = `/tmp/.${outputPath.split("/").pop()}.lerobot-cache`;
const fixturePath = "/tmp/lerobot-dataconvert-pool-ui-fixture";
await rm(outputPath, { recursive: true, force: true });
await rm(cachePath, { recursive: true, force: true });
await rm(fixturePath, { recursive: true, force: true });
execFileSync(
  python,
  ["-c", `from pathlib import Path; from tests.test_multiprocessing_pool_adapter import create_pool_dataset; create_pool_dataset(Path("${fixturePath}"), episodes=2, frames=8, action_pattern=[0,0,0,1,1,1,1,2])`],
  { cwd: new URL("..", import.meta.url).pathname, stdio: "inherit" },
);

const browser = await chromium.launch({ headless: true, executablePath: "/usr/bin/google-chrome" });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const errors = [];

async function assertCreateButtonInRailViewport(label) {
  const metrics = await page.locator("#createJobButton").evaluate((button) => {
    const buttonBox = button.getBoundingClientRect();
    const railBox = button.closest(".control-rail").getBoundingClientRect();
    return {
      buttonTop: buttonBox.top,
      buttonBottom: buttonBox.bottom,
      railTop: railBox.top,
      railBottom: railBox.bottom,
      viewportBottom: window.innerHeight,
      hidden: button.hidden,
      display: getComputedStyle(button).display,
    };
  });
  if (
    metrics.hidden
    || metrics.display === "none"
    || metrics.buttonTop < metrics.railTop - 1
    || metrics.buttonBottom > Math.min(metrics.railBottom, metrics.viewportBottom) + 1
  ) throw new Error(`Start conversion button escaped the control rail (${label}): ${JSON.stringify(metrics)}`);
  return metrics;
}

page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
page.on("pageerror", (error) => errors.push(error.message));
page.on("response", (response) => {
  if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`);
});

await page.goto(baseURL, { waitUntil: "networkidle" });
await page.waitForFunction(() => document.querySelector("#runtimeLabel")?.textContent?.includes("0.1.0"));
await page.waitForFunction(
  () => document.querySelector("#updateNotice")?.dataset.status && document.querySelector("#updateNotice").dataset.status !== "checking",
  null,
  { timeout: 65000 },
);
const updateStatus = await page.getAttribute("#updateNotice", "data-status");
if (await page.locator("#updateCheck").isDisabled()) throw new Error("Manual update check is unavailable");
if ((await page.locator("#updatePull").isHidden()) === (updateStatus === "update_available")) throw new Error(`Pull visibility does not match ${updateStatus}`);
if (updateStatus === "local_changes") {
  const updateNotice = await page.textContent("#updateNotice");
  if (!updateNotice.includes("暂停") || !updateNotice.includes("技术帮助") || !updateNotice.includes("Agent")) throw new Error("Local-change update guidance is incomplete");
}
await assertCreateButtonInRailViewport("desktop update notice");
await page.setViewportSize({ width: 900, height: 600 });
await page.waitForTimeout(50);
await assertCreateButtonInRailViewport("short desktop viewport");
await page.setViewportSize({ width: 1440, height: 1000 });
await page.waitForTimeout(50);
await page.screenshot({ path: "/tmp/lerobot-ui-desktop-update.png" });
await page.click('[data-browse="sourcePath"]');
await page.waitForSelector("#directoryDialog[open] .directory-entry");
await page.click('#directoryDialog button[value="cancel"]');
await page.selectOption("#adapterSelect", "multiprocessing_pool_dataset");
await page.fill("#sourcePath", fixturePath);
await page.fill("#outputPath", outputPath);
await page.fill("#repoId", "ui-e2e");
await page.fill("#robotType", "test_arm");
await page.fill("#taskInstruction", "Move the test arm through the recorded trajectory.");
if ((await page.getAttribute("#cpuLimitPercent", "max")) !== "95") throw new Error("CPU limit can exceed 95 percent");
if (await page.locator("#fillZeroStateAction").isDisabled()) throw new Error("Zero state/action fill is disabled before inspection");
await page.locator("#cpuLimitPercent").fill("95");
await page.click("#inspectButton");
await page.waitForSelector("#datasetReadout:not([hidden])");
const fpsInput = page.locator('[data-adapter-option="fps"]');
if ((await fpsInput.inputValue()) !== "20" || (await fpsInput.getAttribute("max")) !== "20") throw new Error("Target FPS and field-rate limit were not applied");
if (!(await page.textContent("#datasetReadout")).includes("OUTPUT / MAX FPS")) throw new Error("Output FPS summary did not render");
if ((await page.locator(".field-map-row").count()) !== 0) throw new Error("Field mappings were populated automatically");
await page.click("[data-field-map-add]");
const sampleMapping = page.locator(".field-map-row").last();
if ((await sampleMapping.locator("[data-map-source] option").count()) !== 8) throw new Error("Raw field choices did not render");
await sampleMapping.locator("[data-map-source]").selectOption("joint_state/qpos");
if (!(await sampleMapping.locator("[data-map-meta]").textContent()).includes("20.00 FPS")) throw new Error("Per-field FPS did not render in the source selector");
if ((await sampleMapping.locator("[data-map-target]").inputValue()) !== "") throw new Error("Field target was populated automatically");
const declaredFields = await page.textContent("#motionActionFields");
if (!declaredFields.includes("joint_state/qpos + joint_state/qvel + joint_state/torque")) throw new Error("Declared state fields did not render");
if (!declaredFields.includes("joint_action/action + eef_action/action")) throw new Error("Declared action fields did not render");
await fpsInput.fill("10");
await fpsInput.press("Tab");
if (!(await page.locator("#createJobButton").isDisabled())) throw new Error("Changing target FPS did not require a rescan");
await fpsInput.fill("20");
await page.click("#inspectButton");
await page.waitForSelector("#createJobButton:not([disabled])");
const mappings = [
  ["joint_state/qpos", "observation.state"],
  ["joint_state/qpos", "observation.qpos_copy"],
  ["joint_state/qvel", "observation.velocity"],
  ["joint_action/action", "action"],
  ["Cam1", "observation.images.head"],
  ["Cam2", "observation.images.left_wrist"],
];
for (const [source, target] of mappings) {
  await page.click("[data-field-map-add]");
  const row = page.locator(".field-map-row").last();
  await row.locator("[data-map-source]").selectOption(source);
  await row.locator("[data-map-target]").fill(target);
}
if ((await page.locator('[data-map-source] option:checked').filter({ hasText: "joint_state/qpos" }).count()) !== 2) throw new Error("Duplicate raw fields were not selectable");
const buttonBeforeRailScroll = await assertCreateButtonInRailViewport("long field mapping");
await page.locator(".rail-scroll").evaluate((element) => { element.scrollTop = Math.floor(element.scrollHeight / 2); });
const buttonAfterRailScroll = await assertCreateButtonInRailViewport("scrolled field mapping");
if (Math.abs(buttonBeforeRailScroll.buttonTop - buttonAfterRailScroll.buttonTop) > 1) throw new Error("Start conversion footer moved with rail content");
await page.locator(".field-map-row").first().evaluate((element) => element.scrollIntoView({ block: "center" }));
await page.screenshot({ path: "/tmp/lerobot-ui-desktop-fields.png" });
await page.check("#fillZeroStateAction");
await page.check("#trimStationaryStart");
await page.check("#removeStationarySegments");
await page.locator("#stationaryFrames").fill("3");
await page.click("#motionScanButton");
await page.waitForSelector("#motionScanReadout:not([hidden])");
const motionScan = await page.textContent("#motionScanReadout");
if (!motionScan.includes("12 / 16 FR") || !motionScan.includes("0.60 s")) throw new Error(`Unexpected motion scan: ${motionScan}`);
if ((await page.locator("#motionScanReadout dd").first().textContent()) !== "4") throw new Error("Motion segment total is incorrect");
await page.screenshot({ path: "/tmp/lerobot-ui-motion-scan.png" });
await page.locator("#videoCrf").fill("22");
if ((await page.textContent("#videoCrfOutput")) !== "CRF 22") throw new Error("Video CRF output did not update");
await page.locator("#videoCrf").evaluate((element) => element.scrollIntoView({ block: "center" }));
await page.screenshot({ path: "/tmp/lerobot-ui-video-compression.png" });
await page.locator('label:has(input[name="revision"][value="v3.0"])').click();
await page.click("#createJobButton");
await page.waitForSelector("#jobsBody tr");
await page.waitForFunction(() => document.querySelector("#detailState")?.textContent?.includes("已完成"), null, { timeout: 60000 });
await page.waitForFunction(() => document.querySelector("#outputPreview")?.classList.contains("loaded"), null, { timeout: 15000 });
if ((await page.textContent("#detailName")) !== "ui-e2e") throw new Error("Job form values changed during bootstrap");
execFileSync(
  python,
  ["-c", `import pyarrow.parquet as pq; table = pq.read_table("${outputPath}/data/chunk-000/file-000.parquet"); assert "observation.velocity" in table.column_names; assert "observation.qpos_copy" in table.column_names; assert table.num_rows == 4, table.num_rows`],
  { stdio: "inherit" },
);
await page.waitForTimeout(4800);

const audit = await page.evaluate(() => ({
  width: document.documentElement.scrollWidth,
  viewport: window.innerWidth,
  icons: document.querySelectorAll("svg.lucide").length,
  unnamedButtons: [...document.querySelectorAll("button")].filter((button) => !button.textContent.trim() && !button.getAttribute("aria-label") && !button.title).length,
  rawLoaded: document.querySelector("#rawPreview").classList.contains("loaded"),
  outputLoaded: document.querySelector("#outputPreview").classList.contains("loaded"),
  serviceWorker: "serviceWorker" in navigator,
}));
if (audit.width > audit.viewport + 1) throw new Error(`Desktop body overflows: ${audit.width} > ${audit.viewport}`);
if (audit.icons < 8) throw new Error(`Lucide icons did not render: ${audit.icons}`);
if (audit.unnamedButtons) throw new Error(`Unnamed icon buttons: ${audit.unnamedButtons}`);
if (!audit.rawLoaded || !audit.outputLoaded) throw new Error("Preview images did not render");

await page.screenshot({ path: "/tmp/lerobot-ui-desktop.png", fullPage: true });
await page.setViewportSize({ width: 390, height: 844 });
await page.evaluate(() => window.scrollTo(0, 0));
await page.screenshot({ path: "/tmp/lerobot-ui-mobile-control.png" });
await page.locator("#fieldMapping").scrollIntoViewIfNeeded();
await page.screenshot({ path: "/tmp/lerobot-ui-mobile-fields.png" });
await page.locator("#motionScanReadout").scrollIntoViewIfNeeded();
await page.screenshot({ path: "/tmp/lerobot-ui-mobile-motion.png" });
await page.locator("#videoCrf").scrollIntoViewIfNeeded();
await page.screenshot({ path: "/tmp/lerobot-ui-mobile-compression.png" });
await page.locator(".jobs-band").scrollIntoViewIfNeeded();
await page.screenshot({ path: "/tmp/lerobot-ui-mobile-jobs.png" });
const mobileWidth = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: window.innerWidth }));
if (mobileWidth.width > mobileWidth.viewport + 1) throw new Error(`Mobile body overflows: ${mobileWidth.width} > ${mobileWidth.viewport}`);

const jobRow = page.locator("#jobsBody tr").filter({ hasText: "ui-e2e" });
const jobId = await jobRow.getAttribute("data-job-id");
const jobRecord = await (await page.request.get(`${baseURL}/api/jobs/${jobId}`)).json();
if (!Array.isArray(jobRecord.config.field_mapping)) throw new Error("Field mappings were not persisted as rows");
if (jobRecord.config.field_mapping.filter((row) => row.source === "joint_state/qpos").length !== 2) throw new Error("Duplicate raw field mappings were not persisted");
if (!jobRecord.config.field_mapping.some((row) => row.source === "joint_state/qvel" && row.target === "observation.velocity")) throw new Error("Field mapping was not persisted");
if (jobRecord.config.video_crf !== 22) throw new Error("Video CRF was not persisted");
if (jobRecord.config.cpu_limit_percent !== 95) throw new Error("CPU utilization limit was not persisted");
if (!jobRecord.config.fill_zero_state_action) throw new Error("Zero state/action fill was not persisted");
if (!jobRecord.config.trim_stationary_start || !jobRecord.config.remove_stationary_segments || jobRecord.config.stationary_frames !== 3) throw new Error("Motion rules were not persisted");
if (jobRecord.removed_frames !== 12 || jobRecord.removed_segments !== 4) throw new Error("Converted motion filtering differs from pre-scan");
await jobRow.locator('[data-job-action="delete"]').click();
await page.waitForFunction(
  (deletedJobId) => !document.querySelector(`#jobsBody tr[data-job-id="${deletedJobId}"]`),
  jobId,
);
const jobsAfterDelete = await (await page.request.get(`${baseURL}/api/jobs`)).json();
if (jobsAfterDelete.jobs.some((job) => job.id === jobId)) throw new Error("Deleted task remains in the list");
execFileSync(
  python,
  ["-c", `from pathlib import Path; assert Path("${outputPath}/meta/info.json").is_file(); assert Path("${cachePath}/manifest.json").is_file()`],
  { stdio: "inherit" },
);
await page.evaluate(() => navigator.serviceWorker.ready);
await page.context().setOffline(true);
await page.reload({ waitUntil: "domcontentloaded" });
await page.waitForSelector(".brand-block h1");
if ((await page.textContent(".brand-block h1")) !== "LEROBOT DATA CONVERT") throw new Error("PWA shell did not load offline");
await page.waitForSelector("#backendNotice:not([hidden])");
const backendNotice = await page.textContent("#backendNotice");
if (!backendNotice.includes("INSTALL.md") || !backendNotice.includes("systemctl --user restart lerobot-dataconvert")) throw new Error("Backend recovery guidance is incomplete");
await page.locator("#backendNotice").scrollIntoViewIfNeeded();
await page.screenshot({ path: "/tmp/lerobot-ui-mobile-backend-offline.png" });
await page.click("#backendRetry");
await page.waitForFunction(() => !document.querySelector("#backendRetry")?.disabled && document.querySelector("#backendNotice")?.hidden === false);
await page.context().setOffline(false);
await page.waitForFunction(() => document.querySelector("#backendNotice")?.hidden === true);
await browser.close();
await rm(outputPath, { recursive: true, force: true });
await rm(cachePath, { recursive: true, force: true });
await rm(fixturePath, { recursive: true, force: true });

const unexpectedErrors = errors.filter((message) => !message.includes("ERR_INTERNET_DISCONNECTED"));
if (unexpectedErrors.length) throw new Error(`Browser errors:\n${unexpectedErrors.join("\n")}`);
console.log(JSON.stringify(audit));
