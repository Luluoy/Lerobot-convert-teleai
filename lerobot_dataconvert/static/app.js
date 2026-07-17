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
  bootstrapBusy: false,
  updateBusy: false,
  updateStatus: null,
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

const updateStatusLabels = {
  checking: "正在检查远端",
  local_changes: "本地含有修改",
  update_available: "发现远端更新",
  up_to_date: "代码已是最新",
  ahead: "本地领先远端",
  diverged: "Git 历史已分叉",
  unavailable: "自动更新不可用",
  error: "更新检查失败",
  updated: "更新已拉取",
  not_checked: "尚未检查更新",
};

function invalidateInspection() {
  state.descriptor = null;
  $("#createJobButton").disabled = true;
  $("#motionActionFields").hidden = true;
  clearMotionScan();
  updateMotionControls();
}

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

function renderRepositoryUpdate(status) {
  state.updateStatus = status;
  const notice = $("#updateNotice");
  notice.hidden = false;
  notice.dataset.status = status.status;
  $("#updateTitle").textContent = updateStatusLabels[status.status] || "代码更新";
  $("#updateMessage").textContent = status.message || "--";
  const meta = [status.branch, status.upstream];
  if (status.local_change_count) meta.push(`${status.local_change_count} LOCAL CHANGE${status.local_change_count === 1 ? "" : "S"}`);
  if (Number(status.ahead)) meta.push(`AHEAD ${status.ahead}`);
  if (Number(status.behind)) meta.push(`BEHIND ${status.behind}`);
  if (status.checked_at) meta.push(new Date(status.checked_at * 1000).toLocaleString("zh-CN", { hour12: false }));
  const metadata = $("#updateMeta");
  metadata.textContent = meta.filter(Boolean).join(" / ");
  metadata.hidden = !metadata.textContent;
  $("#updatePull").hidden = status.status !== "update_available";
  window.lucide?.createIcons();
}

async function checkRepositoryUpdates(manual = false) {
  if (state.updateBusy) return;
  state.updateBusy = true;
  renderRepositoryUpdate({ status: "checking", message: "正在检查本地状态与远端提交。" });
  $("#updateCheck").disabled = true;
  $("#updatePull").disabled = true;
  try {
    renderRepositoryUpdate(await request("/api/update/check", { method: "POST", body: { manual } }));
  } catch (error) {
    renderRepositoryUpdate({ status: "error", message: `无法检查远端更新：${error.message}` });
  } finally {
    state.updateBusy = false;
    $("#updateCheck").disabled = false;
    $("#updatePull").disabled = false;
  }
}

async function pullRepositoryUpdate() {
  if (state.updateBusy) return;
  state.updateBusy = true;
  $("#updateCheck").disabled = true;
  $("#updatePull").disabled = true;
  try {
    const status = await request("/api/update/pull", { method: "POST", body: {} });
    renderRepositoryUpdate(status);
    toast(status.status === "updated" ? "UPDATED" : "UPDATE", status.message);
  } catch (error) {
    toast("UPDATE FAILED", error.message);
  } finally {
    state.updateBusy = false;
    $("#updateCheck").disabled = false;
    $("#updatePull").disabled = false;
  }
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
  updateMotionControls();
  if (state.jobs.length) selectJob(state.jobs[0].id, false);
  window.lucide?.createIcons();
  checkRepositoryUpdates(false);
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
    if (option.step != null) input.step = option.step;
    if (option.placeholder) input.placeholder = option.placeholder;
    input.addEventListener("change", invalidateInspection);
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

function declaredActionFields(descriptor) {
  return descriptorFields(descriptor).filter((field) => field.is_action);
}

function declaredStateFields(descriptor) {
  return descriptorFields(descriptor).filter((field) => field.is_state);
}

function motionRules() {
  return {
    fill_zero_state_action: $("#fillZeroStateAction").checked,
    trim_stationary_start: $("#trimStationaryStart").checked,
    remove_stationary_segments: $("#removeStationarySegments").checked,
    stationary_frames: Number($("#stationaryFrames").value),
  };
}

function clearMotionScan() {
  const readout = $("#motionScanReadout");
  readout.hidden = true;
  readout.replaceChildren();
}

function configureMotionFields(descriptor) {
  const stateFields = declaredStateFields(descriptor);
  const actionFields = declaredActionFields(descriptor);
  const readout = $("#motionActionFields");
  readout.hidden = false;
  readout.querySelector("[data-state-fields]").textContent = stateFields.length ? stateFields.map((field) => field.name).join(" + ") : "未声明";
  readout.querySelector("[data-action-fields]").textContent = actionFields.length ? actionFields.map((field) => field.name).join(" + ") : "未声明";
  clearMotionScan();
  updateMotionControls();
}

function updateMotionControls() {
  const removeSegments = $("#removeStationarySegments").checked;
  $("#stationaryFrames").disabled = !removeSegments;
  $("#stationaryFramesField").classList.toggle("disabled", !removeSegments);
  const fps = Number(state.descriptor?.fps || adapterOptions().fps || 20);
  const frames = Number($("#stationaryFrames").value);
  $("#stationaryFramesOutput").textContent = `${frames} FR / ${formatNumber(frames / fps, 2)} s`;
  const hasActions = state.descriptor && declaredActionFields(state.descriptor).length > 0;
  $("#motionScanButton").disabled = !hasActions || (!$("#trimStationaryStart").checked && !removeSegments);
}

async function scanMotion() {
  if (!state.descriptor) return toast("MOTION", "请先扫描原始数据");
  const button = $("#motionScanButton");
  button.disabled = true;
  button.querySelector("span").textContent = "扫描中";
  try {
    const result = await request("/api/motion-scan", {
      method: "POST",
      body: {
        adapter: $("#adapterSelect").value,
        source_path: $("#sourcePath").value,
        adapter_options: adapterOptions(),
        ...motionRules(),
      },
    });
    const readout = $("#motionScanReadout");
    readout.hidden = false;
    readout.innerHTML = `<dl>
      <dt>SEGMENTS</dt><dd>${formatInteger(result.segments)}</dd>
      <dt>START SEG</dt><dd>${formatInteger(result.leading_segments)}</dd>
      <dt>STILL SEG</dt><dd>${formatInteger(result.stationary_segments)}</dd>
      <dt>REMOVED</dt><dd>${formatInteger(result.removed_frames)} / ${formatInteger(result.source_frames)} FR</dd>
      <dt>TIME</dt><dd>${formatNumber(result.removed_seconds, 2)} s</dd>
      <dt>KEPT</dt><dd>${formatInteger(result.kept_frames)} FR</dd>
    </dl>`;
    savePreferences();
  } finally {
    button.querySelector("span").textContent = "预扫描";
    updateMotionControls();
  }
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
    const fpsInput = $('[data-adapter-option="fps"]');
    if (fpsInput) {
      fpsInput.max = String(Math.max(1, Math.floor(Number(descriptor.max_output_fps || descriptor.fps) + 1e-9)));
      fpsInput.value = String(descriptor.fps);
    }
    renderDescriptor(descriptor);
    configureMotionFields(descriptor);
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
  const fields = descriptorFields(descriptor);
  const imageCount = fields.filter((field) => field.is_image).length;
  const actionCount = declaredActionFields(descriptor).length;
  const readout = $("#datasetReadout");
  readout.hidden = false;
  readout.innerHTML = `<dl>
    <dt>EPISODES</dt><dd>${descriptor.episodes.length}</dd>
    <dt>FRAMES</dt><dd>${formatInteger(descriptor.total_frames)}</dd>
    <dt>CAMERAS</dt><dd>${descriptor.cameras.length}</dd>
    <dt>FIELDS / IMG</dt><dd>${fields.length} / ${imageCount}</dd>
    <dt>ACTION FIELDS</dt><dd>${actionCount}</dd>
    <dt>STATE / ACTION</dt><dd>${descriptor.state_dim} / ${descriptor.action_dim}</dd>
    <dt>OUTPUT / MAX FPS</dt><dd>${formatNumber(descriptor.fps, 0)} / ${formatNumber(descriptor.max_output_fps || descriptor.fps, 2)}</dd>
    <dt>SOURCE</dt><dd>${formatBytes(descriptor.source_bytes)}</dd>
    <dt>WORKER EST.</dt><dd>${formatInteger(descriptor.estimated_worker_memory_mb)} MiB</dd>
  </dl>${descriptor.warnings.length ? `<p>${escapeHtml(descriptor.warnings[0])}</p>` : ""}`;

  const mapping = $("#fieldMapping");
  const targetSuggestions = [...new Set([
    "observation.state",
    "action",
    "observation.velocity",
    "observation.effort",
    "observation.eef_pose",
    ...fields.map((field) => field.default_target).filter(Boolean),
  ])];
  mapping.hidden = false;
  mapping.innerHTML = `<div class="field-map-toolbar">
      <span class="field-label">Field mapping</span>
      <button class="icon-button field-map-add" type="button" data-field-map-add aria-label="添加字段映射" title="添加字段映射"><i data-lucide="plus"></i></button>
    </div>
    <datalist id="lerobotFieldTargets">${targetSuggestions.map((target) => `<option value="${escapeHtml(target)}"></option>`).join("")}</datalist>
    <div class="field-map-head"><span>RAW FIELD</span><span></span><span>LEROBOT FIELD</span><span></span></div>
    <div class="field-map-rows" id="fieldMappingRows"><p class="field-map-empty">暂无字段映射</p></div>
    <label class="field-label" for="stateNames">observation.state names override</label>
    <textarea id="stateNames" rows="2" placeholder="state_0, state_1, ..."></textarea>
    <label class="field-label" for="actionNames">action names override</label>
    <textarea id="actionNames" rows="2" placeholder="action_0, action_1, ..."></textarea>`;
  window.lucide?.createIcons();
}

function fieldSummary(field) {
  const roles = [field.is_state && "STATE", field.is_action && "ACTION", field.is_image && "IMG"].filter(Boolean);
  const kind = roles.length ? roles.join("+") : "DATA";
  return `${kind} / ${field.dtype.toUpperCase()} / ${(field.shape || []).join("x")} / ${formatNumber(field.fps || state.descriptor?.fps, 2)} FPS`;
}

function addFieldMappingRow(source = "", target = "") {
  if (!state.descriptor) return;
  const fields = descriptorFields(state.descriptor);
  const row = document.createElement("div");
  row.className = "field-map-row";
  row.innerHTML = `<div class="field-map-source">
      <select data-map-source required aria-label="原始数据字段">
        <option value="">选择原始字段</option>
        ${fields.map((field) => `<option value="${escapeHtml(field.name)}">${escapeHtml(field.name)}</option>`).join("")}
      </select>
      <span data-map-meta>--</span>
    </div>
    <i data-lucide="arrow-right" aria-hidden="true"></i>
    <input type="text" data-map-target list="lerobotFieldTargets" required placeholder="observation..." aria-label="LeRobot 目标字段">
    <button class="icon-button field-map-remove" type="button" data-field-map-remove aria-label="删除字段映射" title="删除字段映射"><i data-lucide="x"></i></button>`;
  row.querySelector("[data-map-source]").value = source;
  row.querySelector("[data-map-target]").value = target;
  $("#fieldMappingRows").append(row);
  updateFieldMappingRow(row);
  window.lucide?.createIcons();
}

function updateFieldMappingRow(row) {
  const source = row.querySelector("[data-map-source]").value;
  const field = descriptorFields(state.descriptor).find((item) => item.name === source);
  row.querySelector("[data-map-meta]").textContent = field ? `${field.name} / ${fieldSummary(field)}` : "--";
}

function collectFieldMappings() {
  const rows = $$(".field-map-row", $("#fieldMappingRows")).map((row) => ({
    source: row.querySelector("[data-map-source]").value.trim(),
    target: row.querySelector("[data-map-target]").value.trim(),
  }));
  if (!rows.length) throw new Error("请至少添加一条字段映射");
  if (rows.some((row) => !row.source || !row.target)) throw new Error("每条字段映射都必须选择源字段并填写目标字段");
  const targets = rows.map((row) => row.target);
  if (new Set(targets).size !== targets.length) throw new Error("LeRobot 目标字段不能重复");
  return rows;
}

function configFieldMappingRows(mapping) {
  if (Array.isArray(mapping)) return mapping;
  return Object.entries(mapping || {}).map(([source, target]) => ({ source, target }));
}

async function createJob(event) {
  event.preventDefault();
  if (!state.descriptor) return toast("SOURCE", "请先扫描原始数据");
  const fieldMapping = collectFieldMappings();
  const button = $("#createJobButton");
  button.disabled = true;
  try {
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
      cpu_limit_percent: Number($("#cpuLimitPercent").value),
      memory_gb: Number($("#memoryGb").value),
      segment_size: Number($("#segmentSize").value),
      video_crf: Number($("#videoCrf").value),
      ...motionRules(),
      field_mapping: fieldMapping,
      state_names: splitNames($("#stateNames")?.value),
      action_names: splitNames($("#actionNames")?.value),
      adapter_options: adapterOptions(),
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
  if (state.pollBusy) return $("#backendNotice").hidden;
  state.pollBusy = true;
  try {
    const data = await request("/api/jobs");
    state.jobs = data.jobs;
    setConnection(true);
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
    return true;
  } catch (_) {
    setConnection(false);
    return false;
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
    const canDelete = !["queued", "running", "merging", "stopping"].includes(job.state);
    row.innerHTML = `
      <td><strong class="job-name">${escapeHtml(job.name)}</strong><span class="job-path">${escapeHtml(job.config.output_path)}</span></td>
      <td><span class="state-label">${statusLabels[job.state] || escapeHtml(job.state)}</span></td>
      <td><div class="row-progress"><div class="row-progress-track"><span style="width:${progress}%"></span></div><output>${progress}%</output></div></td>
      <td>${formatFinish(job.eta_seconds, job.state)}</td>
      <td>${formatBytes(job.estimated_output_bytes)}</td>
      <td><div class="job-actions">${canStop ? actionButton("square", "stop", "中断任务") : ""}${canResume ? actionButton("play", "resume", "继续任务") : ""}${canDelete ? actionButton("trash-2", "delete", "从列表删除（保留本地文件）") : ""}</div></td>`;
    row.addEventListener("click", (event) => {
      if (!event.target.closest("[data-job-action]")) selectJob(job.id);
    });
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
  const action = button.dataset.jobAction;
  button.disabled = true;
  try {
    if (action === "delete") {
      await request(`/api/jobs/${jobId}`, { method: "DELETE" });
      if (state.selectedJobId === jobId) {
        state.selectedJobId = null;
        state.previewDescriptor = null;
        $("#detailBand").hidden = true;
        $("#previewBand").hidden = true;
      }
      toast("REMOVED", "任务已从列表移除，本地文件未更改");
    } else {
      await request(`/api/jobs/${jobId}/${action}`, { method: "POST", body: {} });
    }
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
  $("#detailState").textContent = `${statusLabels[job.state] || job.state} / ${job.config.revision} / CRF ${job.config.video_crf ?? 30} / CPU MAX ${job.config.cpu_limit_percent ?? 95}%`;
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
  const camera = $("#previewCamera");
  const previous = camera.value;
  let cameras = descriptorFields(descriptor).filter((field) => field.is_image).map((field) => field.name);
  if (job?.config?.field_mapping) {
    const mappedSources = new Set(configFieldMappingRows(job.config.field_mapping).map((row) => row.source));
    cameras = cameras.filter((name) => mappedSources.has(name));
  }
  $("#previewBand").hidden = cameras.length === 0;
  camera.replaceChildren(...cameras.map((name) => new Option(name, name)));
  if (!cameras.length) return;
  if (cameras.includes(previous)) camera.value = previous;
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
  if (!$("#previewCamera").options.length) return;
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
  $("#cpuLimitOutput").textContent = `${$("#cpuLimitPercent").value}% MAX`;
  $("#memoryOutput").textContent = `${$("#memoryGb").value} GiB`;
  $("#segmentOutput").textContent = `${$("#segmentSize").value} EP`;
  $("#videoCrfOutput").textContent = `CRF ${$("#videoCrf").value}`;
  updateMotionControls();
}

function savePreferences() {
  localStorage.setItem("lerobot-dataconvert-form", JSON.stringify({
    source_path: $("#sourcePath").value,
    output_path: $("#outputPath").value,
    repo_id: $("#repoId").value,
    robot_type: $("#robotType").value,
    task_instruction: $("#taskInstruction").value,
    cpu_cores: $("#cpuCores").value,
    cpu_limit_percent: $("#cpuLimitPercent").value,
    memory_gb: $("#memoryGb").value,
    segment_size: $("#segmentSize").value,
    video_crf: $("#videoCrf").value,
    fill_zero_state_action: $("#fillZeroStateAction").checked,
    trim_stationary_start: $("#trimStationaryStart").checked,
    remove_stationary_segments: $("#removeStationarySegments").checked,
    stationary_frames: $("#stationaryFrames").value,
  }));
}

function restorePreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem("lerobot-dataconvert-form"));
    if (!saved) return;
    for (const [key, value] of Object.entries(saved)) {
      const input = $(`#${camelId(key)}`);
      if (input?.type === "checkbox") input.checked = Boolean(value);
      else if (input && value != null) input.value = value;
    }
    updateResourceOutputs();
  } catch (_) { /* ignore invalid local preference data */ }
}

function selectedJob() { return state.jobs.find((job) => job.id === state.selectedJobId); }
function splitNames(value = "") { return value.split(/[,\n]/).map((item) => item.trim()).filter(Boolean); }
function descriptorFields(descriptor) {
  if (descriptor?.fields?.length) return descriptor.fields;
  return [
    { name: "state", shape: [descriptor.state_dim], dtype: "float32", is_image: false, is_state: true, default_target: "observation.state", fps: descriptor.fps },
    { name: "action", shape: [descriptor.action_dim], dtype: "float32", is_image: false, is_action: true, default_target: "action", fps: descriptor.fps },
    ...descriptor.cameras.map((camera) => ({
      name: camera,
      shape: descriptor.camera_shapes[camera],
      dtype: "uint8",
      is_image: true,
      fps: descriptor.fps,
      default_target: `observation.images.${defaultCameraName(camera)}`,
    })),
  ];
}
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
  $("#backendNotice").hidden = online;
}

async function initializeBackend(showFailure = false) {
  if (state.bootstrap) return true;
  if (state.bootstrapBusy) return false;
  state.bootstrapBusy = true;
  try {
    await bootstrap();
    setConnection(true);
    return true;
  } catch (error) {
    setConnection(false);
    if (showFailure) toast("SERVER", error.message);
    return false;
  } finally {
    state.bootstrapBusy = false;
  }
}

async function checkBackend() {
  const button = $("#backendRetry");
  button.disabled = true;
  try {
    const online = state.bootstrap ? await refreshJobs() : await initializeBackend(true);
    if (!online && state.bootstrap) toast("SERVER", "后端仍不可用，请检查或重启服务");
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  $("#adapterSelect").addEventListener("change", () => { renderAdapterOptions(); invalidateInspection(); });
  $("#sourcePath").addEventListener("change", invalidateInspection);
  $("#inspectButton").addEventListener("click", () => inspectSource().catch((error) => toast("SCAN FAILED", error.message)));
  $("#motionScanButton").addEventListener("click", () => scanMotion().catch((error) => toast("MOTION SCAN", error.message)));
  $("#jobForm").addEventListener("submit", (event) => createJob(event).catch((error) => toast("CREATE FAILED", error.message)));
  $("#resumePathButton").addEventListener("click", () => resumeFromPath().catch((error) => toast("RECOVERY FAILED", error.message)));
  $("#refreshButton").addEventListener("click", () => state.bootstrap ? refreshJobs() : initializeBackend(true));
  $("#backendRetry").addEventListener("click", () => checkBackend());
  $("#updateCheck").addEventListener("click", () => checkRepositoryUpdates(true));
  $("#updatePull").addEventListener("click", () => pullRepositoryUpdate());
  $("#jobsBody").addEventListener("click", (event) => handleJobAction(event).catch((error) => toast("TASK", error.message)));
  $("#fieldMapping").addEventListener("click", (event) => {
    if (event.target.closest("[data-field-map-add]")) addFieldMappingRow();
    const remove = event.target.closest("[data-field-map-remove]");
    if (remove) remove.closest(".field-map-row").remove();
  });
  $("#fieldMapping").addEventListener("change", (event) => {
    const source = event.target.closest("[data-map-source]");
    if (source) updateFieldMappingRow(source.closest(".field-map-row"));
  });
  ["#cpuCores", "#cpuLimitPercent", "#memoryGb", "#segmentSize", "#videoCrf"].forEach((selector) => $(selector).addEventListener("input", updateResourceOutputs));
  ["#fillZeroStateAction", "#trimStationaryStart", "#removeStationarySegments"].forEach((selector) => $(selector).addEventListener("change", () => { clearMotionScan(); updateMotionControls(); }));
  $("#stationaryFrames").addEventListener("input", () => { clearMotionScan(); updateMotionControls(); });
  $$('[data-browse]').forEach((button) => button.addEventListener("click", () => openPicker(button.dataset.browse).catch((error) => toast("PATH", error.message))));
  $("#directoryUp").addEventListener("click", async () => renderDirectory(await request(`/api/fs?path=${encodeURIComponent($("#directoryUp").dataset.path)}`)));
  $("#selectDirectory").addEventListener("click", () => choosePickerValue(state.pickerPath));
  $("#episodeSlider").addEventListener("input", () => { $("#frameSlider").value = 0; updateTimelineOutputs(); schedulePreview(); });
  $("#frameSlider").addEventListener("input", () => { updateTimelineOutputs(); schedulePreview(); });
  $("#previewCamera").addEventListener("change", schedulePreview);
  $("#reloadPreview").addEventListener("click", () => loadPreview());
  window.addEventListener("online", () => state.bootstrap ? refreshJobs() : initializeBackend());
  window.addEventListener("offline", () => setConnection(false));
  window.addEventListener("beforeinstallprompt", (event) => { event.preventDefault(); state.installPrompt = event; $("#installButton").hidden = false; });
  $("#installButton").addEventListener("click", async () => { if (state.installPrompt) await state.installPrompt.prompt(); });
}

bindEvents();
window.lucide?.createIcons();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js");
initializeBackend();
setInterval(() => { if (state.bootstrap) refreshJobs(); }, 1000);
setInterval(() => { if (!state.bootstrap) initializeBackend(); }, 3000);
