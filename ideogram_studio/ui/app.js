(function () {
  const state = {
    config: null,
    activeJob: null,
    pollTimer: null,
    helperApi: null,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function apiReady() {
    return window.pywebview && window.pywebview.api;
  }

  async function callApi(method, ...args) {
    if (!apiReady()) {
      throw new Error("Desktop bridge is not ready yet.");
    }
    return window.pywebview.api[method](...args);
  }

  function setStatus(text, isError) {
    const node = $("status-line");
    node.textContent = text || "";
    node.classList.toggle("error", !!isError);
  }

  function setProgress(label, percent) {
    $("progress-label").textContent = label || "Idle";
    $("progress-fill").style.width = `${Math.max(0, Math.min(100, Number(percent) || 0))}%`;
  }

  function fillSelect(select, items, valueKey) {
    select.innerHTML = "";
    items.forEach((item) => {
      const option = document.createElement("option");
      if (typeof item === "string") {
        option.value = item;
        option.textContent = item;
      } else {
        option.value = item.value;
        option.textContent = item.label;
      }
      select.appendChild(option);
    });
    if (valueKey && state.config && state.config[valueKey]) {
      select.value = state.config[valueKey];
    }
  }

  function showPreviewFromB64(b64) {
    const img = $("preview-image");
    const empty = document.querySelector(".preview-empty");
    if (!b64) return;
    img.src = `data:image/png;base64,${b64}`;
    img.hidden = false;
    if (empty) empty.hidden = true;
  }

  async function reloadLayoutHelper(sanitize) {
    const shouldSanitize = sanitize === true;
    const raw = $("prompt-json").value.trim();
    if (shouldSanitize && raw && apiReady()) {
      try {
        const fixed = await callApi("sanitize_prompt", raw);
        if (fixed.ok && fixed.pretty) {
          $("prompt-json").value = fixed.pretty;
        }
      } catch (_err) {
        /* keep existing JSON if sanitize fails */
      }
    }
    if (state.helperApi) state.helperApi.open();
  }

  async function applyLayoutChanges() {
    const active = document.activeElement;
    if (active && active.closest(".ideogram4-helper-list, .ideogram4-helper-box-editor, .ideogram4-helper-selected")) {
      active.blur();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (state.helperApi && typeof state.helperApi.apply === "function") {
      state.helperApi.apply();
    } else {
      const applyBtn = document.querySelector('[data-ideogram4-action="apply"]');
      if (applyBtn) applyBtn.click();
    }
    await new Promise((resolve) => setTimeout(resolve, 40));
  }

  function initPromptHelper() {
    window.wangpConfirm = window.wangpConfirm || function (opts) {
      const message = (opts && (opts.message || opts.title)) || "Confirm?";
      return Promise.resolve(window.confirm(message));
    };
    const card = document.querySelector("[data-ideogram4-prompt-helper]");
    if (!card || !window.wangpIdeogram4PromptHelper) return;
    state.helperApi = window.wangpIdeogram4PromptHelper.init(card);
    if (state.helperApi) state.helperApi.open();
  }

  async function bootstrap() {
    while (!apiReady()) {
      await new Promise((resolve) => setTimeout(resolve, 120));
    }
    const payload = await callApi("get_bootstrap");
    state.config = payload.config || {};
    fillSelect($("aspect-ratio"), payload.aspect_ratios || [], "aspect_ratio");
    fillSelect($("resolution-select"), payload.resolutions || [], "resolution");
    fillSelect($("model-type"), payload.models || [], "model_type");
    fillSelect($("model-mode"), payload.presets || [], "model_mode");
    fillSelect($("cfg-model"), payload.deepseek_models || [], "deepseek_model");
    $("cfg-thinking").checked = !!state.config.deepseek_thinking;
    $("cfg-profile").value = state.config.wgp_profile || "";
    $("cfg-output").value = state.config.output_dir || "";
    $("engine-status").textContent = "Engine warming up…";
    initPromptHelper();
    setTimeout(() => {
      $("engine-status").textContent = "Engine ready";
    }, 1500);
  }

  async function runMagicPrompt() {
    $("magic-btn").disabled = true;
    setStatus("Calling DeepSeek…");
    try {
      const result = await callApi(
        "magic_prompt",
        $("idea-text").value,
        $("aspect-ratio").value
      );
      if (!result.ok) throw new Error(result.error || "Magic Prompt failed.");
      $("prompt-json").value = result.pretty || result.json || "";
      reloadLayoutHelper(true);
      const notes = (result.sanitize_notes || []).join("; ");
      setStatus(notes ? `Magic Prompt applied (${notes}).` : "Magic Prompt applied — layout synced.");
    } catch (err) {
      setStatus(err.message || String(err), true);
    } finally {
      $("magic-btn").disabled = false;
    }
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function pollJob(jobId) {
    const snapshot = await callApi("poll_job", jobId);
    if (snapshot.error && !snapshot.done) {
      setStatus(snapshot.error, true);
      stopPolling();
      $("generate-btn").disabled = false;
      $("cancel-btn").disabled = true;
      return;
    }
    setProgress(snapshot.status || snapshot.phase || "Working…", snapshot.progress);
    if (snapshot.preview_b64) showPreviewFromB64(snapshot.preview_b64);
    if (snapshot.done) {
      stopPolling();
      $("generate-btn").disabled = false;
      $("cancel-btn").disabled = true;
      if (snapshot.success) {
        const path = (snapshot.output_paths || [])[0] || "";
        $("output-path").textContent = path;
        if (snapshot.preview_b64) {
          showPreviewFromB64(snapshot.preview_b64);
        }
        setStatus("Done.");
        setProgress("Complete", 100);
      } else {
        setStatus(snapshot.error || "Generation failed.", true);
        setProgress("Failed", snapshot.progress || 0);
      }
    }
  }

  async function runGenerate() {
    await applyLayoutChanges();
    const prompt = $("prompt-json").value.trim();
    if (!prompt) {
      setStatus("Prompt JSON is empty.", true);
      return;
    }
    $("generate-btn").disabled = true;
    $("cancel-btn").disabled = false;
    setStatus("Queued…");
    setProgress("Starting…", 0);
    try {
      const result = await callApi(
        "start_generate",
        prompt,
        $("resolution-select").value,
        $("model-type").value,
        $("model-mode").value,
        Number($("seed-input").value || -1)
      );
      if (!result.ok) throw new Error(result.error || "Could not start generation.");
      state.activeJob = result.job_id;
      stopPolling();
      state.pollTimer = setInterval(() => pollJob(state.activeJob), 350);
      await pollJob(state.activeJob);
    } catch (err) {
      setStatus(err.message || String(err), true);
      $("generate-btn").disabled = false;
      $("cancel-btn").disabled = true;
    }
  }

  async function cancelGenerate() {
    if (!state.activeJob) return;
    await callApi("cancel_job", state.activeJob);
    setStatus("Cancel requested…");
  }

  async function saveSettings() {
    const payload = {
      deepseek_api_key: $("cfg-api-key").value.trim(),
      deepseek_model: $("cfg-model").value,
      deepseek_thinking: $("cfg-thinking").checked,
      wgp_profile: $("cfg-profile").value.trim(),
      output_dir: $("cfg-output").value.trim(),
      model_type: $("model-type").value,
      model_mode: $("model-mode").value,
      resolution: $("resolution-select").value,
      aspect_ratio: $("aspect-ratio").value,
    };
    const result = await callApi("save_config", payload);
    if (result.ok) {
      state.config = result.config;
      setStatus("Settings saved.");
      $("settings-drawer").hidden = true;
      $("settings-toggle").setAttribute("aria-expanded", "false");
    }
  }

  $("magic-btn").addEventListener("click", runMagicPrompt);
  $("generate-btn").addEventListener("click", runGenerate);
  $("cancel-btn").addEventListener("click", cancelGenerate);
  $("save-settings").addEventListener("click", saveSettings);
  $("settings-toggle").addEventListener("click", () => {
    const drawer = $("settings-drawer");
    const open = drawer.hidden;
    drawer.hidden = !open;
    $("settings-toggle").setAttribute("aria-expanded", open ? "true" : "false");
  });

  $("resolution-select").addEventListener("change", () => reloadLayoutHelper(false));
  $("sync-layout-btn").addEventListener("click", () => reloadLayoutHelper(true));

  $("prompt-json").addEventListener("blur", () => reloadLayoutHelper(false));

  window.addEventListener("pywebviewready", bootstrap);
  if (apiReady()) bootstrap();
})();
