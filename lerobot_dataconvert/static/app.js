const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  bootstrap: null,
  jobs: [],
  descriptor: null,
  selectedJobId: null,
  previewDescriptor: null,
  pickerTarget: null,
  pickerPath: null,
  installPrompt: null,
  previewTimer: null,
  pollBusy: false,
  lastOutputSegmentCount: -1,
};

const statusLabels = {
  queued: "等待",
  running: "转换中",
  merging: "合并中",
  stopping: "停止中",
  paused: "已中断",
  completed: "已完成",
  failed: "失败",
  canceled: "已取消",
};

async function request(path, options = {}, expectBlob = false) {
  const init = { ...options, headers: { ...(options.headers || {}) } };
  if (init.body && typeof init.body !== "string") {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(init.body);
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).error || message; } catch (_) { /* response is not JSON */ }
    throw new Error(message);
  }
  return expectBlob ? response.blob() : response.json();
}

async function bootstrap() {
  const data = await request("/api/bootstrap");
  state.bootstrap = data;
  state.jobs = data.jobs;
  $("#runtimeLabel").textContent = `LOCAL WORKBENCH / ${data.version}`;
  renderAdapters();
  renderRevisions();
  configureResources(data.hardware);
  renderJobs();
  restorePreferences();
  if (state.jobs.length) selectJob(state.jobs[0].id, false);
  window.lucide?.createIcons();
}

function renderAdapters() {
  const select = $("#adapterSelect");
  select.replaceChildren(...state.bootstrap.adapters.map((adapter) => new Option(adapter.name, adapter.slug)));
  renderAdapterOptions();
}

function renderAdapterOptions() {
  const adapter = state.bootstrap.adapters.find((item) => item.slug === $("#adapterSelect").value);
  const root = $("#adapterOptions");
  root.replaceChildren();
  for (const option of adapter?.options || []) {
    const label = document.createElement("label");
    label.className = "field-label";
    label.htmlFor = `adapter-option-${option.key}`;
    label.textContent = option.label;
    const input = document.createElement("input");
    input.id = `adapter-option-${option.key}`;
    input.dataset.adapterOption = option.key;
    input.type = option.type || "text";
    input.value = option.default ?? "";
    if (option.min != null) input.min = option.min;
    if (option.max != null) input.max = option.max;
    root.append(label, input);
  }
}

function renderRevisions() {
  const root = $("#revisionControl");
  root.replaceChildren();
  state.bootstrap.revisions.forEach((revision, index) => {
    const label = document.createElement("label");
    label.title = revision.description;
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "revision";
    input.value = revision.id;
    input.checked = index === 0;
    label.append(input, document.createTextNode(revision.id));
    root.append(label);
  });
}

function configureResources(hardware) {
  const cpu = $("#cpuCores");
  cpu.max = hardware.cpu_count;
  cpu.value = Math.min(6, hardware.cpu_count);
  const memory = $("#memoryGb");
  memory.max = Math.max(2, Math.floor(hardware.memory_total_gb));
  memory.value = Math.min(8, Math.max(2, Math.floor(hardware.memory_available_gb / 2)));
  $("#globalCpu").textContent = `${hardware.cpu_count} CORE`;
  $("#globalMemory").textContent = `${formatNumber(hardware.memory_available_gb, 1)} GiB`;
  updateResourceOutputs();
}

function adapterOptions() {
  const output = {};
  $$('[data-adapter-option]').forEach((input) => {
    output[input.dataset.adapterOption] = input.type === "number" ? Number(input.value) : input.value;
  });
  return output;
}

async function inspectSource() {
  const button = $("#inspectButton");
  button.disabled = true;
  button.querySelector("span").textContent = "扫描中";
  try {
    const descriptor = await request("/api/inspect", {
      method: "POST",
      body: {
        adapter: $("#adapterSelect").value,
        source_path: $("#sourcePath").value,
        adapter_options: adapterOptions(),
      },
    });
    state.descriptor = descriptor;
    state.previewDescriptor = descriptor;
    renderDescriptor(descriptor);
    $("#createJobButton").disabled = false;
    if (!$("#repoId").value) $("#repoId").value = basename($("#outputPath").value) || basename(descriptor.source_path);
    configurePreview(descriptor, null);
    await loadPreview();
    savePreferences();
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "扫描数据";
  }
}

function renderDescriptor(descriptor) {
  const readout = $("#datasetReadout");
  readout.hidden = false;
  readout.innerHTML = `<dl>
    <dt>EPISODES</dt><dd>${descriptor.episodes.length}</dd>
    <dt>FRAMES</dt><dd>${formatInteger(descriptor.total_frames)}</dd>
    <dt>CAMERAS</dt><dd>${descriptor.cameras.length}</dd>
    <dt>STATE / ACTION</dt><dd>${descriptor.state_dim} / ${descriptor.action_dim}</dd>
    <dt>SOURCE</dt><dd>${formatBytes(descriptor.source_bytes)}</dd>
    <dt>WORKER EST.</dt><dd>${formatInteger(descriptor.estimated_worker_memory_mb)} MiB</dd>
  </dl>${descriptor.warnings.length ? `<p>${escapeHtml(descriptor.warnings[0])}</p>` : ""}`;

  const mapping = $("#cameraMapping");
  mapping.hidden = false;
  mapping.innerHTML = `<span class="field-label">Camera output names</span>${descriptor.cameras.map((camera) => `
    <div class="camera-row"><label title="${escapeHtml(camera)}">${escapeHtml(camera)}</label>
    <input data-camera-name="${escapeHtml(camera)}" value="${escapeHtml(defaultCameraName(camera))}" aria-label="${escapeHtml(camera)} 输出名称"></div>`).join("")}
    <label class="field-label" for="stateNames">State names</label>
    <textarea id="stateNames" rows="2" placeholder="state_0, state_1, ..."></textarea>
    <label class="field-label" for="actionNames">Action names</label>
    <textarea id="actionNames" rows="2" placeholder="action_0, action_1, ..."></textarea>`;
}

async function createJob(event) {
  event.preventDefault();
  if (!state.descriptor) return toast("SOURCE", "请先扫描原始数据");
  const button = $("#createJobButton");
  button.disabled = true;
  try {
    const cameraNames = {};
    $$('[data-camera-name]').forEach((input) => { cameraNames[input.dataset.cameraName] = input.value; });
    const payload = {
      adapter: $("#adapterSelect").value,
      source_path: $("#sourcePath").value,
      output_path: $("#outputPath").value,
      revision: $('input[name="revision"]:checked').value,
      repo_id: $("#repoId").value,
      robot_type: $("#robotType").value,
      task_instruction: $("#taskInstruction").value,
      fps: Number(adapterOptions().fps || state.descriptor.fps),
      cpu_cores: Number($("#cpuCores").value),
      memory_gb: Number($("#memoryGb").value),
      segment_size: Number($("#segmentSize").value),
      camera_names: cameraNames,
      state_names: splitNames($("#stateNames")?.value),
      action_names: splitNames($("#actionNames")?.value),
      adapter_options: adapterOptions(),
      skip_zero_state: $("#skipZeroState").checked,
      overwrite: $("#overwriteOutput").checked,
    };
    const job = await request("/api/jobs", { method: "POST", body: payload });
    savePreferences();
    await refreshJobs();
    selectJob(job.id);
    toast("QUEUED", `${job.name} 已加入任务队列`);
  } finally {
    button.disabled = false;
  }
}

async function resumeFromPath() {
  const outputPath = $("#outputPath").value.trim();
  if (!outputPath) return toast("PATH", "请输入保存路径");
  const job = await request("/api/jobs/resume-path", { method: "POST", body: { output_path: outputPath } });
  await refreshJobs();
  selectJob(job.id);
  toast("RECOVERED", "任务已从 sidecar cache 恢复");
}

async function refreshJobs() {
  if (state.pollBusy) return;
  state.pollBusy = true;
  try {
    const data = await request("/api/jobs");
    state.jobs = data.jobs;
    renderJobs();
    if (state.selectedJobId) {
      const job = selectedJob();
      if (job) {
        renderDetail(job);
        if (job.completed_segments !== state.lastOutputSegmentCount) {
          state.lastOutputSegmentCount = job.completed_segments;
          if (job.completed_segments > 0 || job.state === "completed") loadPreview("output");
        }
      }
    }
  } catch (_) {
    setConnection(false);
  } finally {
    state.pollBusy = false;
  }
}

function renderJobs() {
  const body = $("#jobsBody");
  body.replaceChildren();
  $("#jobsEmpty").hidden = state.jobs.length > 0;
  $("#jobCount").textContent = `${state.jobs.length} TASK${state.jobs.length === 1 ? "" : "S"}`;
  $("#activeJobs").textContent = state.jobs.filter((job) => ["running", "merging", "stopping"].includes(job.state)).length;
  $("#queuedJobs").textContent = state.jobs.filter((job) => job.state === "queued").length;
  $("#doneJobs").textContent = state.jobs.filter((job) => job.state === "completed").length;

  for (const job of state.jobs) {
    const row = document.createElement("tr");
    row.dataset.jobId = job.id;
    if (job.id === state.selectedJobId) row.classList.add("selected");
    const progress = Math.round((job.progress || 0) * 100);
    const canStop = ["queued", "running", "merging"].includes(job.state);
    const canResume = ["paused", "failed"].includes(job.state);
    row.innerHTML = `
      <td><strong class="job-name">${escapeHtml(job.name)}</strong><span class="job-path">${escapeHtml(job.config.output_path)}</span></td>
      <td><span class="state-label">${statusLabels[job.state] || escapeHtml(job.state)}</span></td>
      <td><div class="row-progress"><div class="row-progress-track"><span style="width:${progress}%"></span></div><output>${progress}%</output></div></td>
      <td>${formatFinish(job.eta_seconds, job.state)}</td>
      <td>${formatBytes(job.estimated_output_bytes)}</td>
      <td><div class="job-actions">${canStop ? actionButton("square", "stop", "中断任务") : ""}${canResume ? actionButton("play", "resume", "继续任务") : ""}</div></td>`;
    row.addEventListener("click", () => selectJob(job.id));
    body.append(row);
  }
  window.lucide?.createIcons();
}

function actionButton(icon, action, label) {
  return `<button class="icon-button" type="button" data-job-action="${action}" aria-label="${label}" title="${label}"><i data-lucide="${icon}"></i></button>`;
}

async function handleJobAction(event) {
  const button = event.target.closest("[data-job-action]");
  if (!button) return;
  event.stopPropagation();
  const jobId = button.closest("tr").dataset.jobId;
  button.disabled = true;
  try {
    await request(`/api/jobs/${jobId}/${button.dataset.jobAction}`, { method: "POST", body: {} });
    await refreshJobs();
  } finally {
    button.disabled = false;
  }
}

function selectJob(jobId, reload = true) {
  state.selectedJobId = jobId;
  state.lastOutputSegmentCount = -1;
  const job = selectedJob();
  if (!job) return;
  renderJobs();
  renderDetail(job);
  state.previewDescriptor = job.descriptor;
  configurePreview(job.descriptor, job);
  if (reload) {
    if (job.completed_segments > 0 || job.state === "completed") {
      loadPreview();
    } else {
      clearImage($("#outputPreview"), "等待已完成 segment");
      loadPreview("raw");
    }
  }
}

function renderDetail(job) {
  $("#detailBand").hidden = false;
  $("#detailState").textContent = `${statusLabels[job.state] || job.state} / ${job.config.revision}`;
  $("#detailName").textContent = job.name;
  $("#detailPath").textContent = job.config.output_path;
  const percent = Math.round((job.progress || 0) * 100);
  $("#detailPercent").innerHTML = `${percent}<span>%</span>`;
  $("#detailProgress").style.width = `${percent}%`;
  $("#metricEpisodes").textContent = `${job.completed_episodes} / ${job.total_episodes}`;
  $("#metricFrames").textContent = `${formatInteger(job.completed_frames + job.active_frames)} / ${formatInteger(job.total_frames)}`;
  $("#metricElapsed").textContent = formatDuration(job.elapsed_seconds);
  $("#metricEta").textContent = job.eta_seconds == null ? "--" : `${formatDuration(job.eta_seconds)} / ${formatFinish(job.eta_seconds, job.state)}`;
  $("#metricSize").textContent = `${formatBytes(job.written_bytes)} / ${formatBytes(job.estimated_output_bytes)}`;
  $("#metricResources").textContent = `${job.effective_workers} / ${formatNumber(job.memory_rss_mb, 0)} MiB`;
  $("#detailMessage").textContent = job.error || job.message || "--";
}

function configurePreview(descriptor, job) {
  $("#previewBand").hidden = false;
  const camera = $("#previewCamera");
  const previous = camera.value;
  camera.replaceChildren(...descriptor.cameras.map((name) => new Option(name, name)));
  if (descriptor.cameras.includes(previous)) camera.value = previous;
  const episode = $("#episodeSlider");
  episode.max = Math.max(0, descriptor.episodes.length - 1);
  episode.value = Math.min(Number(episode.value), Number(episode.max));
  updateTimelineOutputs();
  $("#outputCaption").textContent = job ? job.config.revision : "--";
}

function updateTimelineOutputs() {
  const descriptor = state.previewDescriptor;
  if (!descriptor?.episodes.length) return;
  const episodeIndex = Number($("#episodeSlider").value);
  const episode = descriptor.episodes[episodeIndex];
  const frame = $("#frameSlider");
  frame.max = Math.max(0, episode.frame_count - 1);
  frame.value = Math.min(Number(frame.value), Number(frame.max));
  $("#episodeOutput").textContent = `${episodeIndex} / ${descriptor.episodes.length - 1}`;
  $("#frameOutput").textContent = `${frame.value} / ${frame.max}`;
  $("#rawCaption").textContent = episode.key;
}

function schedulePreview() {
  clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(() => loadPreview(), 160);
}

async function loadPreview(only = "both") {
  const descriptor = state.previewDescriptor;
  if (!descriptor?.episodes.length) return;
  updateTimelineOutputs();
  const episode = Number($("#episodeSlider").value);
  const frame = Number($("#frameSlider").value);
  const camera = $("#previewCamera").value;
  const job = selectedJob();

  if (only !== "output") {
    try {
      let blob;
      if (job) {
        blob = await request(`/api/jobs/${job.id}/preview?kind=raw&episode=${episode}&frame=${frame}&camera=${encodeURIComponent(camera)}&t=${Date.now()}`, {}, true);
      } else {
        blob = await request("/api/preview/raw", {
          method: "POST",
          body: {
            adapter: $("#adapterSelect").value,
            source_path: $("#sourcePath").value,
            adapter_options: adapterOptions(),
            episode: descriptor.episodes[episode],
            camera,
            frame_index: frame,
          },
        }, true);
      }
      setBlobImage($("#rawPreview"), blob);
    } catch (error) {
      clearImage($("#rawPreview"), error.message);
    }
  }

  if (only !== "raw") {
    if (!job) return clearImage($("#outputPreview"), "等待任务");
    try {
      const blob = await request(`/api/jobs/${job.id}/preview?kind=output&episode=${episode}&frame=${frame}&camera=${encodeURIComponent(camera)}&t=${Date.now()}`, {}, true);
      setBlobImage($("#outputPreview"), blob);
    } catch (error) {
      clearImage($("#outputPreview"), error.message);
    }
  }
}

function setBlobImage(image, blob) {
  if (image.dataset.objectUrl) URL.revokeObjectURL(image.dataset.objectUrl);
  const url = URL.createObjectURL(blob);
  image.dataset.objectUrl = url;
  image.onload = () => image.classList.add("loaded");
  image.src = url;
}

function clearImage(image, message) {
  image.classList.remove("loaded");
  image.removeAttribute("src");
  image.nextElementSibling.textContent = message;
}

async function openPicker(targetId) {
  state.pickerTarget = targetId;
  const input = $(`#${targetId}`);
  let initial = input.value.trim() || (targetId === "sourcePath" ? "/media" : "/home/amin");
  try {
    const data = await request(`/api/fs?path=${encodeURIComponent(initial)}`);
    renderDirectory(data);
  } catch (_) {
    const parent = initial.includes("/") ? initial.slice(0, initial.lastIndexOf("/")) || "/" : "/";
    renderDirectory(await request(`/api/fs?path=${encodeURIComponent(parent)}`));
  }
  $("#directoryDialog").showModal();
  window.lucide?.createIcons();
}

function renderDirectory(data) {
  state.pickerPath = data.path;
  $("#directoryPath").textContent = data.path;
  $("#directoryUp").dataset.path = data.parent;
  $("#directoryFree").textContent = `${formatBytes(data.free_bytes)} FREE`;
  const list = $("#directoryList");
  list.replaceChildren();
  for (const entry of data.entries) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "directory-entry";
    button.dataset.path = entry.path;
    button.dataset.kind = entry.kind;
    button.innerHTML = `<i data-lucide="${entry.kind === "directory" ? "folder" : "file"}"></i><span>${escapeHtml(entry.name)}</span><small>${entry.kind === "file" ? formatBytes(entry.size) : "DIR"}</small>`;
    button.addEventListener("click", async () => {
      if (entry.kind === "directory") renderDirectory(await request(`/api/fs?path=${encodeURIComponent(entry.path)}`));
      else if (state.pickerTarget === "sourcePath") choosePickerValue(entry.path);
      window.lucide?.createIcons();
    });
    list.append(button);
  }
  window.lucide?.createIcons();
}

function choosePickerValue(value) {
  const input = $(`#${state.pickerTarget}`);
  input.value = value;
  input.dispatchEvent(new Event("change"));
  $("#directoryDialog").close();
}

function updateResourceOutputs() {
  $("#cpuOutput").textContent = `${$("#cpuCores").value} CORE`;
  $("#memoryOutput").textContent = `${$("#memoryGb").value} GiB`;
  $("#segmentOutput").textContent = `${$("#segmentSize").value} EP`;
}

function savePreferences() {
  localStorage.setItem("lerobot-dataconvert-form", JSON.stringify({
    source_path: $("#sourcePath").value,
    output_path: $("#outputPath").value,
    repo_id: $("#repoId").value,
    robot_type: $("#robotType").value,
    task_instruction: $("#taskInstruction").value,
    cpu_cores: $("#cpuCores").value,
    memory_gb: $("#memoryGb").value,
    segment_size: $("#segmentSize").value,
  }));
}

function restorePreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem("lerobot-dataconvert-form"));
    if (!saved) return;
    for (const [key, value] of Object.entries(saved)) {
      const input = $(`#${camelId(key)}`);
      if (input && value != null) input.value = value;
    }
    updateResourceOutputs();
  } catch (_) { /* ignore invalid local preference data */ }
}

function selectedJob() { return state.jobs.find((job) => job.id === state.selectedJobId); }
function splitNames(value = "") { return value.split(/[,\n]/).map((item) => item.trim()).filter(Boolean); }
function basename(path) { return path?.replace(/\/+$/, "").split("/").pop() || ""; }
function defaultCameraName(camera) { const match = camera.match(/(\d+)$/); return match ? `image_${match[1]}` : camera.replace(/[^a-zA-Z0-9_]+/g, "_").toLowerCase(); }
function camelId(value) { return value.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase()); }
function formatInteger(value) { return Number(value || 0).toLocaleString("zh-CN"); }
function formatNumber(value, digits = 0) { return Number(value || 0).toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits }); }
function formatBytes(value) {
  if (value == null || !Number.isFinite(Number(value)) || Number(value) <= 0) return "--";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(Math.floor(Math.log(Number(value)) / Math.log(1024)), units.length - 1);
  return `${formatNumber(Number(value) / 1024 ** index, index > 2 ? 1 : 0)} ${units[index]}`;
}
function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "--";
  const total = Math.max(0, Math.round(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const rest = total % 60;
  return hours ? `${hours}h ${String(minutes).padStart(2, "0")}m` : `${minutes}m ${String(rest).padStart(2, "0")}s`;
}
function formatFinish(seconds, jobState) {
  if (jobState === "completed") return "DONE";
  if (seconds == null || !Number.isFinite(Number(seconds))) return "--";
  const date = new Date(Date.now() + Number(seconds) * 1000);
  return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false });
}
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character])); }

function toast(title, message) {
  const item = document.createElement("div");
  item.className = "toast";
  item.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(message)}</span>`;
  $("#toastRegion").append(item);
  setTimeout(() => item.remove(), 4600);
}

function setConnection(online) {
  const node = $("#connectionState");
  node.classList.toggle("offline", !online);
  node.lastChild.textContent = online ? "LOCAL" : "OFFLINE";
}

function bindEvents() {
  $("#adapterSelect").addEventListener("change", () => { renderAdapterOptions(); state.descriptor = null; $("#createJobButton").disabled = true; });
  $("#sourcePath").addEventListener("change", () => { state.descriptor = null; $("#createJobButton").disabled = true; });
  $("#inspectButton").addEventListener("click", () => inspectSource().catch((error) => toast("SCAN FAILED", error.message)));
  $("#jobForm").addEventListener("submit", (event) => createJob(event).catch((error) => toast("CREATE FAILED", error.message)));
  $("#resumePathButton").addEventListener("click", () => resumeFromPath().catch((error) => toast("RECOVERY FAILED", error.message)));
  $("#refreshButton").addEventListener("click", () => refreshJobs());
  $("#jobsBody").addEventListener("click", (event) => handleJobAction(event).catch((error) => toast("TASK", error.message)));
  ["#cpuCores", "#memoryGb", "#segmentSize"].forEach((selector) => $(selector).addEventListener("input", updateResourceOutputs));
  $$('[data-browse]').forEach((button) => button.addEventListener("click", () => openPicker(button.dataset.browse).catch((error) => toast("PATH", error.message))));
  $("#directoryUp").addEventListener("click", async () => renderDirectory(await request(`/api/fs?path=${encodeURIComponent($("#directoryUp").dataset.path)}`)));
  $("#selectDirectory").addEventListener("click", () => choosePickerValue(state.pickerPath));
  $("#episodeSlider").addEventListener("input", () => { $("#frameSlider").value = 0; updateTimelineOutputs(); schedulePreview(); });
  $("#frameSlider").addEventListener("input", () => { updateTimelineOutputs(); schedulePreview(); });
  $("#previewCamera").addEventListener("change", schedulePreview);
  $("#reloadPreview").addEventListener("click", () => loadPreview());
  window.addEventListener("online", () => { setConnection(true); refreshJobs(); });
  window.addEventListener("offline", () => setConnection(false));
  window.addEventListener("beforeinstallprompt", (event) => { event.preventDefault(); state.installPrompt = event; $("#installButton").hidden = false; });
  $("#installButton").addEventListener("click", async () => { if (state.installPrompt) await state.installPrompt.prompt(); });
}

bindEvents();
bootstrap().then(() => {
  setConnection(true);
  setInterval(refreshJobs, 1000);
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
}).catch((error) => {
  setConnection(false);
  toast("SERVER", error.message);
});
