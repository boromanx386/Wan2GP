(function studioHelperPatch() {
  const helper = window.wangpIdeogram4PromptHelper;
  if (!helper || helper._studioPatched) return;
  helper._studioPatched = true;

  const origReadResolution = helper.readResolution;
  helper.readResolution = function (targetId) {
    const target = document.getElementById(targetId || "");
    if (target) {
      if (target.tagName === "SELECT" || target.tagName === "INPUT") {
        const parsed = helper.parseResolutionText(target.value);
        if (parsed) return parsed;
      }
      if (target.value) {
        const parsed = helper.parseResolutionText(target.value);
        if (parsed) return parsed;
      }
    }
    return origReadResolution.call(this, targetId);
  };

  const origInit = helper.init.bind(helper);
  helper.init = function (card) {
    const api = origInit(card);
    if (!api || !card) return api;

    card.classList.add("ideogram-studio-helper");
    const content = card.querySelector(".ideogram4-prompt-helper-content");
    const workspace = card.querySelector(".ideogram4-helper-workspace");
    const details = card.querySelector(".ideogram4-helper-details");
    const splitter = card.querySelector(".ideogram4-helper-splitter");
    const boxEditor = card.querySelector(".ideogram4-helper-box-editor");
    const canvas = card.querySelector("canvas");
    const canvasShell = card.querySelector(".ideogram4-helper-canvas-shell");
    const canvasColumn = card.querySelector(".ideogram4-helper-canvas-column");
    const descField = card.querySelector('[data-ideogram4-field="desc"]');
    const descLabel = descField ? descField.closest("label") : null;

    if (details) {
      const fields = details.querySelector(".ideogram4-helper-fields");
      if (fields) fields.style.display = "none";
      if (splitter) splitter.style.display = "none";
      const selected = details.querySelector(".ideogram4-helper-selected");
      if (selected && canvasColumn) {
        details.style.display = "none";
        canvasColumn.appendChild(selected);
        selected.classList.add("ideogram-studio-selected-panel");
      }
    }

    function showDescForSelectedElement() {
      if (!descLabel) return;
      descLabel.style.display = "";
    }

    if (boxEditor) {
      boxEditor.style.pointerEvents = "auto";
      boxEditor.addEventListener("focus", showDescForSelectedElement);
    }

    card.addEventListener("click", (event) => {
      if (event.target.closest(".ideogram4-helper-list, canvas, .ideogram4-helper-selected")) {
        showDescForSelectedElement();
      }
    });

    if (workspace) {
      workspace.style.gridTemplateColumns = "minmax(0, 1fr) minmax(130px, 190px)";
    }

    if (canvas) {
      canvas.style.touchAction = "none";
    }

    if (canvasShell) {
      canvasShell.style.aspectRatio = "1 / 1";
      canvasShell.style.width = "auto";
      canvasShell.style.height = "100%";
      canvasShell.style.maxWidth = "100%";
      canvasShell.style.maxHeight = "100%";
      canvasShell.style.margin = "0 auto";
    }

    function relayoutCanvas() {
      if (!content || !workspace) return;
      const height = Math.max(380, content.clientHeight - 8);
      workspace.style.setProperty("flex", "1 1 auto", "important");
      workspace.style.setProperty("height", height + "px", "important");
      workspace.style.setProperty("flex-basis", height + "px", "important");
      workspace.style.setProperty("min-height", "380px", "important");
      window.dispatchEvent(new Event("resize"));
    }

    if (content && window.ResizeObserver) {
      const observer = new ResizeObserver(() => relayoutCanvas());
      observer.observe(content);
      card._studioResizeObserver = observer;
    }

    const origOpen = api.open.bind(api);
    return {
      open: function () {
        origOpen();
        requestAnimationFrame(() => {
          relayoutCanvas();
          requestAnimationFrame(relayoutCanvas);
        });
      },
      apply: function () {
        const applyBtn = card.querySelector('[data-ideogram4-action="apply"]');
        if (applyBtn) applyBtn.click();
      },
    };
  };
})();
