import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Local server folders, not browser uploads. No image/metadata leaves ComfyUI.
let dialog;
const ko = (navigator.language || "").startsWith("ko");
const t = (k, e) => ko ? k : e;

function el(tag, text, parent) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (parent) parent.append(node);
    return node;
}

async function request(action, body) {
    const response = await api.fetchApi(`/rs-vrs/folders${action ? `/${action}` : ""}`, action ? {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
    } : {});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);
    return data;
}

async function showFolders() {
    if (dialog?.open) { dialog.focus(); return; }
    dialog?.remove();
    dialog = el("dialog");
    const panel = dialog;
    panel.style.cssText = "width:min(720px,90vw);max-height:85vh;overflow:auto;background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);border:1px solid #777;border-radius:10px;padding:24px;";
    document.body.append(panel);
    el("h2", t("VRS 폴더 기억", "VRS Folder Memory"), panel);
    el("p", t("폴더와 output의 이미지 메타데이터를 읽고, 생성 큐가 빌 때 조금씩 기억합니다. 새 생성이 들어오면 현재 이미지와 저장을 마친 뒤 양보합니다.",
        "Read image metadata from folders and output; index small batches when the generation queue is idle. New generations take priority after the current image and save finish."), panel);
    const field = (label, tag = "input") => {
        const row = el("label", label, panel);
        row.style.cssText = "display:block;margin:12px 0;";
        const input = el(tag, "", row);
        input.style.cssText = "display:block;box-sizing:border-box;width:100%;margin-top:4px;padding:6px;background:var(--comfy-input-bg,#333);color:inherit;";
        return input;
    };
    const collection = field(t("기억 선택", "Collection"), "select");
    const fresh = el("option", t("새 기억 만들기", "Create new collection"), collection);
    fresh.value = "";
    const label = field(t("이름", "Name"));
    label.value = t("내 이미지", "My images");
    const roots = field(t("ComfyUI 서버의 폴더 절대 경로 — 한 줄에 하나, 하위 폴더 포함", "Absolute folders on the ComfyUI server — one per line, recursive"), "textarea");
    roots.rows = 3;
    const output = field(t("ComfyUI output 폴더도 포함", "Include ComfyUI output folder"));
    output.type = "checkbox"; output.checked = true; output.style.width = "auto";
    const model = field(t("VRS 비전용 네이티브 Gemma safetensors", "Native Gemma safetensors for VRS vision"), "select");
    const batch = field(t("큐 작업당 최대 이미지 — 새 생성에는 이미지 단위로 양보", "Maximum images per queue job — yield between images for new generations"));
    batch.type = "number"; batch.min = 1; batch.max = 32; batch.value = 4;
    el("p", t("모델은 별도 준비가 필요합니다. 메타데이터 없는 파일은 집계만 하며 자동 태깅하지 않습니다. 원본은 수정하지 않습니다. 서버 재시작 후에는 직접 재개하세요.",
        "Prepare the vision model separately. Files without readable metadata are counted, not auto-tagged. Originals stay unchanged. Resume manually after a server restart."), panel);
    const buttons = el("div", "", panel);
    buttons.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
    const start = el("button", t("시작 / 재개", "Start / Resume"), buttons);
    const stop = el("button", t("중지", "Stop"), buttons);
    const add = el("button", t("조회 노드 추가", "Add memory node"), buttons);
    const close = el("button", t("닫기", "Close"), buttons);
    const status = el("pre", "", panel);
    status.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;";
    let state;
    let initialized = false;
    let requestVersion = 0;
    let busy = false;
    const controls = () => {
        start.disabled = busy || !state || state.active || !!state.ticket || state.state === "stopping";
        stop.disabled = busy || !state || (!state.active && !state.ticket);
        add.disabled = !state || !collection.value;
    };
    const editable = () => {
        const selected = state?.collections.find(c => c.id === collection.value);
        if (selected) {
            label.value = selected.label;
            roots.value = selected.roots.join("\n");
            model.value = selected.model_name;
            batch.value = selected.batch_size;
        }
        for (const input of [label, roots, output, model, batch]) input.disabled = !!collection.value;
        controls();
    };
    collection.onchange = editable;
    const refresh = async () => {
        const version = ++requestVersion;
        const response = await request();
        if (version !== requestVersion || !panel.open) return;
        state = response;
        for (const c of state.collections) {
            if (![...collection.options].some(o => o.value === c.id)) {
                const option = el("option", `${c.label} (${c.id.slice(0, 8)})`, collection);
                option.value = c.id;
            }
        }
        const modelNames = [...model.options].map(option => option.value);
        if (JSON.stringify(modelNames) !== JSON.stringify(state.models)) {
            const selected = model.value;
            model.replaceChildren();
            for (const name of state.models) el("option", name, model).value = name;
            if (state.models.includes(selected)) model.value = selected;
            else model.value = state.models.find(name => /gemma/i.test(name)) || state.models[0] || "";
        }
        if (!initialized) {
            collection.value = state.collection?.config.id || (state.collections.length === 1 ? state.collections[0].id : "");
            initialized = true;
            editable();
        }
        const c = state.collection;
        const names = {
            stopped: t("중지됨", "Stopped"), stopping: t("현재 이미지 저장 후 중지", "Stopping after current image"),
            scanning: t("폴더 확인 중", "Scanning"), watching: t("새 파일 대기", "Watching for changes"),
            queued: t("소량 구축 대기", "Batch queued"), encoding: t("이미지 특징 저장 중", "Encoding"),
            "preparing-search": t("검색 준비 중 — 이전 완성 판본 사용 가능", "Preparing search — previous ready snapshot remains available"),
            "waiting-for-generation": t("생성 큐가 빌 때까지 대기", "Waiting for generation queue"), error: t("오류 — 확인 후 재개", "Error — review and resume"),
        };
        status.textContent = `${names[state.state] || state.state}\n${state.error || ""}\n${state.search_error || ""}\n` +
            `${t("output", "Output")}: ${state.output_directory}\n` +
            (c ? `${c.config.label}\n` +
                `${t("캡션 읽음", "Readable metadata")}: ${c.files.included || 0} · ${t("메타데이터 없음", "No metadata")}: ${c.files["no-generation-metadata"] || 0} · ${t("중복", "Duplicates")}: ${c.files["duplicate-bytes"] || 0} · ${t("읽기 오류", "Read errors")}: ${c.files.error || 0}\n` +
                `${t("검색 준비 완료", "Search prepared")}: ${c.search_count || 0} · ${t("경험 저장됨", "Published experiences")}: ${c.records.encoded || 0} · ${t("저장됨·발행 대기", "Saved, awaiting publication")}: ${c.records.staged || 0} · ${t("대기", "Pending")}: ${c.records.pending || 0} · ${t("실패", "Failed")}: ${c.records.error || 0}\n` +
                `Search snapshot: ${c.search_snapshot || "—"}\n` +
                `${t("기억 경로", "Memory path")}: ${c.store_path}\nSnapshot: ${c.snapshot || "—"}\n` +
                (c.errors.length ? JSON.stringify(c.errors, null, 2) : "") : "");
        controls();
    };
    start.onclick = async () => {
        busy = true; ++requestVersion; controls();
        try {
            const result = await request("start", collection.value ? { collection_id: collection.value } : {
                label: label.value, roots: roots.value.split("\n").map(p => p.trim()).filter(Boolean),
                include_output: output.checked, model_name: model.value, batch_size: Number(batch.value),
            });
            await refresh();
            collection.value = result.collection.config.id; editable();
        } catch (error) { status.textContent = error.message; }
        finally { busy = false; controls(); }
    };
    stop.onclick = async () => {
        busy = true; ++requestVersion; controls();
        try { await request("stop"); await refresh(); }
        catch (error) { status.textContent = error.message; }
        finally { busy = false; controls(); }
    };
    add.onclick = () => {
        const id = collection.value;
        if (!id) { status.textContent = t("기억을 먼저 선택하세요.", "Select a collection first."); return; }
        const node = LiteGraph.createNode("VRSFolderMemory");
        if (!node) { status.textContent = t("서버 재시작 후 화면을 새로고침하세요.", "Restart the server, then refresh the page."); return; }
        const widget = node.widgets.find(w => w.name === "collection_id");
        widget.options.values = state.collections.map(c => c.id);
        widget.value = id;
        app.graph.add(node);
        node.pos = [180, 180];
        panel.close();
    };
    close.onclick = () => panel.close();
    controls();
    panel.showModal();
    let timer;
    const poll = async () => {
        if (!panel.open) return;
        if (!busy) {
            try { await refresh(); } catch (error) { if (!busy) status.textContent = error.message; }
        }
        if (panel.open) timer = setTimeout(poll, 2000);
    };
    panel.addEventListener("close", () => clearTimeout(timer), { once: true });
    await poll();
}

app.registerExtension({
    name: "RS.VRS.Folders",
    async setup() {
        const button = document.createElement("button");
        button.textContent = t("VRS 폴더", "VRS Folders");
        button.title = t("개인 이미지 폴더 기억 관리", "Manage personal image folder memory");
        button.style.cssText = "position:fixed;bottom:12px;left:12px;z-index:999;padding:8px 12px;border-radius:6px;background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);border:1px solid #888;";
        button.onclick = showFolders;
        document.body.append(button);
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VRSFolderMemory") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this.addWidget("button", t("VRS 폴더 관리", "Manage VRS Folders"), null, showFolders, { serialize: false });
            return result;
        };
    },
});
