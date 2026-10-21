import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Processing stays on the ComfyUI server. A remote/cloud server is not this PC.
let dialog;
const ko = (navigator.language || "").startsWith("ko");
const t = (k, e) => ko ? k : e;

function el(tag, text, parent) {
    const node = document.createElement(tag);
    if (text) node.textContent = text;
    if (parent) parent.append(node);
    return node;
}

async function request(action, body, selected = "") {
    const query = !action && selected ? `?collection_id=${encodeURIComponent(selected)}` : "";
    const response = await api.fetchApi(`/rs-vrs/folders${action ? `/${action}` : ""}${query}`, action ? {
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
    el("p", t("로컬 처리 전용: 이미지·태그·기억을 외부 저장소로 업로드하거나 동기화하지 않습니다. 본인 PC에서 실행한 ComfyUI를 사용하세요. 원격·클라우드 ComfyUI는 그 서버에서 자료를 처리합니다.",
        "Local processing: no image, tag or memory upload/sync to an external repository. Use ComfyUI running on your own PC; remote/cloud ComfyUI processes data on that server."), panel);
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
    const model = field(t("VRS 비전 — 자동 준비 또는 네이티브 Gemma", "VRS vision — automatic preparation or native Gemma"), "select");
    const batch = field(t("큐 작업당 최대 이미지 — 새 생성에는 이미지 단위로 양보", "Maximum images per queue job — yield between images for new generations"));
    batch.type = "number"; batch.min = 1; batch.max = 32; batch.value = 4;
    const tagger = field(t("메타데이터를 읽을 수 없는 이미지", "Images without usable prompt metadata"), "select");
    el("option", t("건너뛰기 — 추가 모델 없음", "Skip — no extra model"), tagger).value = "none";
    el("option", t("PixAI로 보완 태깅 — 선택 의존성·약 1.81GiB 모델", "PixAI fallback — optional dependencies, ~1.81 GiB model"), tagger).value = "pixai";
    const useCpu = field(t("CPU 추론에 동의 — 느릴 수 있고 다른 작업에 부하가 생깁니다. 기존 GPU 기억과 별도로 만드세요.", "Use CPU inference — may be slow and affect other work. Use a separate collection from GPU memory."));
    useCpu.type = "checkbox"; useCpu.checked = false;
    const resumeOnStart = field(t("서버 재시작 후 이 수집 자동 재개", "Resume this collection after server restart"));
    resumeOnStart.type = "checkbox";
    el("p", t("자동 비전은 첫 큐 작업에서 약 319.3MiB만 받습니다. PixAI는 별도 선택이며 requirements-tagger.txt 설치 후 사용하세요. 최초 약 1.81GiB를 받고 추가 GPU/RAM을 사용합니다. 예측 태그는 원래 프롬프트와 구분하며 원본 파일은 수정하지 않습니다. 다운로드는 중단되면 처음부터 재시도합니다. 자동 재개는 선택 사항이며 중지를 누르면 예약도 해제됩니다.",
        "Auto vision downloads ~319.3 MiB. Optional PixAI requires requirements-tagger.txt and a ~1.81 GiB first-use download plus extra GPU/RAM. Predictions remain separate from original prompts; originals are unchanged. Interrupted downloads restart. Auto-resume is optional; Stop disables it until the next Start."), panel);
    const buttons = el("div", "", panel);
    buttons.style.cssText = "display:flex;gap:8px;flex-wrap:wrap;";
    const check = el("button", t("시작 전 확인", "Check folders & setup"), buttons);
    const start = el("button", t("시작 / 재개", "Start / Resume"), buttons);
    const stop = el("button", t("중지", "Stop"), buttons);
    const add = el("button", t("조회 노드 추가", "Add memory node"), buttons);
    const workflow = el("button", t("VRS → TIPO 연결 추가", "Add VRS → TIPO workflow"), buttons);
    const close = el("button", t("닫기", "Close"), buttons);
    const preview = el("pre", "", panel);
    preview.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;";
    preview.setAttribute("aria-live", "polite");
    const guidance = el("p", "", panel);
    guidance.setAttribute("aria-live", "polite");
    const progress = el("progress", "", panel);
    progress.style.width = "100%";
    progress.hidden = true;
    const status = el("pre", "", panel);
    status.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;";
    let state;
    let initialized = false;
    let requestVersion = 0;
    let busy = false;
    const selectedStatus = () => state?.selected_collection?.config.id === collection.value
        ? state.selected_collection : state?.collection?.config.id === collection.value ? state.collection : null;
    const ready = () => {
        const c = selectedStatus();
        return !!(c && (c.search_snapshot || c.snapshot) && (c.search_count || c.records.encoded));
    };
    const controls = () => {
        start.disabled = busy || !state || state.active || !!state.ticket || state.state === "stopping";
        stop.disabled = busy || !state || (!state.active && !state.ticket);
        check.disabled = busy || !state;
        collection.disabled = busy;
        for (const input of [label, roots, output, model, batch]) input.disabled = busy || !!collection.value;
        add.disabled = workflow.disabled = busy || !ready();
        useCpu.disabled = tagger.disabled = resumeOnStart.disabled = busy || !!state?.active || !!state?.ticket;
    };
    const draftFields = [label, roots, output, model, batch, tagger, useCpu, resumeOnStart];
    const capture = () => draftFields.map(input => ({ value: input.value, checked: input.checked }));
    let newDraft = null;
    let previousCollection = "";
    const editable = () => {
        const selected = state?.collections.find(c => c.id === collection.value);
        if (selected) {
            label.value = selected.label;
            roots.value = selected.roots.join("\n");
            if (![...model.options].some(o => o.value === selected.model_name)) {
                el("option", selected.model_name, model).value = selected.model_name;
            }
            model.value = selected.model_name;
            batch.value = selected.batch_size;
            tagger.value = selected.tagger || "none";
            resumeOnStart.checked = !!selected.resume_on_start;
            useCpu.checked = !!selected.use_cpu;
        } else if (newDraft) draftFields.forEach((input, i) => Object.assign(input, newDraft[i]));
        for (const input of [label, roots, output, model, batch]) input.disabled = !!collection.value;
        controls();
    };
    collection.onchange = async () => {
        if (!previousCollection) newDraft = capture();
        previousCollection = collection.value;
        preview.textContent = "";
        editable();
        try { await refresh(); } catch (error) { status.textContent = error.message; }
    };
    const refresh = async () => {
        const version = ++requestVersion;
        const response = await request("", null, collection.value);
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
            const config = state.collections.find(c => c.id === collection.value);
            if (config && !state.models.includes(config.model_name)) {
                el("option", config.model_name, model).value = config.model_name;
                model.value = config.model_name;
            }
        }
        if (!initialized) {
            newDraft = capture();
            collection.value = state.collection?.config.id || (state.collections.length === 1 ? state.collections[0].id : "");
            previousCollection = collection.value;
            initialized = true;
            editable();
        }
        const c = selectedStatus();
        const encoded = c?.records.encoded || 0;
        const staged = c?.records.staged || 0;
        const pending = (c?.records.pending || 0) + (c?.tagging?.pending || 0);
        const failed = (c?.records.error || 0) + (c?.tagging?.error || 0);
        const total = encoded + staged + pending + failed;
        const importingSelected = state.active && state.collection?.config.id === collection.value;
        progress.hidden = !c || !total;
        progress.max = total || 1;
        progress.value = encoded + staged;
        progress.setAttribute("aria-label", t("현재 발견된 경험의 저장 진행률 — 스캔 중 전체 수는 늘어납니다", "Saved progress among discovered experiences — totals grow during scanning"));
        guidance.textContent = !c ? t("폴더를 입력하고 ‘시작 전 확인’을 눌러보세요. 기본 output 폴더만으로도 시작할 수 있습니다.", "Check your folders and setup first. You can start with just the output folder.")
            : ready() ? t("검색할 준비가 됐습니다. ‘VRS → TIPO 연결 추가’ 후 Load Image에서 이미지를 선택하고 Recall에 요청을 입력하세요.", "Ready to search. Add the VRS → TIPO workflow, choose a Load Image reference and enter your request in Recall.")
            : (pending || staged) && importingSelected ? t("첫 경험을 저장하고 검색을 준비하는 중입니다. 준비가 끝나면 연결 추가 버튼이 활성화됩니다.", "Saving the first experiences and preparing search. Workflow buttons become available when ready.")
            : pending || staged ? t("아직 검색 준비가 끝나지 않았습니다. 이 기억의 수집을 재개해 대기 중인 경험을 처리하세요.", "Search is not ready yet. Resume this collection to process its pending experiences.")
            : importingSelected ? t("수집 중입니다. 폴더를 확인하고 새 이미지를 기다립니다. 메타데이터 없는 이미지는 PixAI를 선택한 경우에만 추가합니다.", "Collection is running: checking folders and waiting for images. Images without prompt metadata are included only with optional PixAI.")
            : t("아직 검색할 경험이 없습니다. 수집을 시작하세요. 읽을 수 있는 메타데이터가 없다면 PixAI 보완 태깅을 선택하세요.", "No searchable experience yet. Start collection; use optional PixAI when prompt metadata is missing.");
        if (c && total) guidance.textContent += `\n${t("발견된 경험 중 저장", "Saved among discovered experiences")}: ${encoded + staged} / ${total}`;
        if (state.active && state.collection?.config.id !== collection.value) {
            guidance.textContent += `\n${t("현재 수집 중인 기억", "Currently importing")}: ${state.collection.config.label}`;
        }
        const names = {
            stopped: t("중지됨", "Stopped"), stopping: t("현재 이미지 저장 후 중지", "Stopping after current image"),
            scanning: t("폴더 확인 중", "Scanning"), watching: t("새 파일 대기", "Watching for changes"),
            queued: t("소량 구축 대기", "Batch queued"), encoding: t("이미지 특징 저장 중", "Encoding"),
            "preparing-vision": t("비전 모델 준비·다운로드 중", "Preparing / downloading vision"),
            "preparing-tagger": t("PixAI 준비·다운로드 중", "Preparing / downloading PixAI"),
            tagging: t("PixAI 보완 태깅 중", "PixAI fallback tagging"),
            "preparing-search": t("검색 준비 중 — 이전 완성 판본 사용 가능", "Preparing search — previous ready snapshot remains available"),
            "waiting-for-generation": t("생성 큐가 빌 때까지 대기", "Waiting for generation queue"), error: t("오류 — 확인 후 재개", "Error — review and resume"),
        };
        status.textContent = `${names[state.state] || state.state}\n${state.error || ""}\n${state.search_error || ""}\n` +
            `${t("output", "Output")}: ${state.output_directory}\n` +
            (c ? `${c.config.label}\n` +
                `${t("캡션 읽음", "Readable metadata")}: ${c.files.included || 0} · ${t("메타데이터 없음", "No metadata")}: ${c.files["no-generation-metadata"] || 0} · ${t("중복", "Duplicates")}: ${c.files["duplicate-bytes"] || 0} · ${t("읽기 오류", "Read errors")}: ${c.files.error || 0}\n` +
                `${t("검색 준비 완료", "Search prepared")}: ${c.search_count || 0} · ${t("경험 저장됨", "Published experiences")}: ${c.records.encoded || 0} · ${t("저장됨·발행 대기", "Saved, awaiting publication")}: ${c.records.staged || 0} · ${t("대기", "Pending")}: ${c.records.pending || 0} · ${t("실패", "Failed")}: ${c.records.error || 0}\n` +
                `PixAI: ${t("대기", "Pending")} ${c.tagging?.pending || 0} · ${t("태깅됨", "Tagged")} ${c.tagging?.tagged || 0} · ${t("확신 태그 없음", "No confident tags")} ${c.tagging?.["no-confident-tags"] || 0} · ${t("실패", "Failed")} ${c.tagging?.error || 0}\n` +
                `Search snapshot: ${c.search_snapshot || "—"}\n` +
                `${t("기억 경로", "Memory path")}: ${c.store_path}\nSnapshot: ${c.snapshot || "—"}\n` +
                (c.errors.length ? JSON.stringify(c.errors, null, 2) : "") : "");
        controls();
    };
    const configuration = () => collection.value ? {
        collection_id: collection.value, tagger: tagger.value, resume_on_start: resumeOnStart.checked, use_cpu: useCpu.checked,
    } : {
        label: label.value, roots: roots.value.split("\n").map(p => p.trim()).filter(Boolean),
        include_output: output.checked, model_name: model.value, batch_size: Number(batch.value), tagger: tagger.value,
        resume_on_start: resumeOnStart.checked, use_cpu: useCpu.checked,
    };
    const checkSetup = async config => {
        preview.textContent = t("로컬 폴더 확인 중…", "Checking local folders…");
        const result = await request("preview", config);
        preview.textContent = t("시작 전 확인 — 일부 파일만 확인한 결과이며 전체 개수·추론 성공을 보장하지 않습니다.", "Setup check — a bounded sample, not a total count or an inference guarantee.") + "\n" +
            result.samples.map(s => `${s.path}\n  ${t("이미지 확인", "Images checked")} ${s.images_checked} · ${t("사용 가능 메타데이터", "Usable metadata")} ${s.readable_metadata} · ${t("메타데이터 없음", "No metadata")} ${s.no_metadata} · ${t("오류", "Errors")} ${s.read_errors}${s.limited ? t(" · 일부만 확인", " · partial sample") : ""}`).join("\n") + "\n" +
            (result.automatic_vision ? t("자동 비전: 로컬 파일 검증 후 필요하면 약 319.3MiB 다운로드.", "Auto vision: verify local files, then download ~319.3 MiB if needed.") : t("선택한 로컬 비전 헤더 확인 완료.", "Selected local vision header checked.")) + "\n" +
            (result.use_cpu ? t("CPU 사용 동의 선택됨. 처리 속도가 느리고 CPU 부하가 생길 수 있습니다.", "CPU consent selected; inference may be slow and CPU intensive.") : t("ComfyUI 가속기 사용. CPU로 자동 전환하지 않습니다.", "Uses the ComfyUI accelerator; no automatic CPU fallback.")) +
            (result.offline ? "\n" + t("오프라인 모드: 필요한 모델이 로컬에 없으면 시작 후 모델 준비가 실패합니다.", "Offline mode: model preparation will fail if required local assets are missing.") : "") +
            (result.missing_packages.length ? `\n${t("필요 패키지 누락", "Missing packages")}: ${result.missing_packages.join(", ")}\n${t("ComfyUI Python으로 requirements-tagger.txt를 설치한 뒤 재시작하거나 PixAI를 끄세요.", "Install requirements-tagger.txt in ComfyUI's Python and restart, or disable PixAI.")}` : "");
        return result;
    };
    check.onclick = async () => {
        busy = true; ++requestVersion; controls();
        try { await checkSetup(configuration()); }
        catch (error) { preview.textContent = error.message; }
        finally { busy = false; controls(); }
    };
    start.onclick = async () => {
        busy = true; ++requestVersion; controls();
        try {
            const config = configuration();
            if (!(await checkSetup(config)).can_start) return;
            const result = await request("start", config);
            const created = result.collection.config;
            if (!collection.value) newDraft = capture();
            if (![...collection.options].some(o => o.value === created.id)) el("option", created.label, collection).value = created.id;
            collection.value = created.id;
            previousCollection = created.id;
            await refresh();
            editable();
        } catch (error) { preview.textContent = error.message; }
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
    workflow.onclick = async () => {
        const config = state.collections.find(c => c.id === collection.value);
        if (!config) return;
        const types = ["VRSFolderMemory", "VRSNativeVisionLoader", "LoadImage", "VRSNativeRecall", "GemmaPromptExpandNative", "PreviewAny", "PreviewAny"];
        const nodes = types.map(name => LiteGraph.createNode(name));
        if (nodes.some(n => !n)) { status.textContent = t("서버·화면을 새로고침하세요.", "Restart server and refresh page."); return; }
        const [memory, vision, image, recall, expand, tags, prose] = nodes;
        const set = (node, name, value) => { const w = node.widgets?.find(w => w.name === name); if (w) w.value = value; };
        memory.widgets.find(w => w.name === "collection_id").options.values = state.collections.map(c => c.id);
        set(memory, "collection_id", config.id);
        set(vision, "model_name", config.model_name);
        set(vision, "use_cpu", !!config.use_cpu);
        set(expand, "model_source", "heretic_ara_q4");
        set(expand, "tipo_lora", "image-30k");
        set(expand, "tipo_lora_scope", "tags_only");
        set(expand, "output_format", "both");
        set(recall, "ratings", "g,s,q,e,u");
        set(recall, "prompt", t("참고 이미지의 인물을 가을 공원에서 산책하는 장면으로 확장해줘.", "Expand the reference character into an autumn park walk."));
        const positions = [[40,40],[40,240],[40,460],[440,40],[900,40],[1400,40],[1400,350]];
        try {
            if (!expand.inputs?.some(x => x.name === "prompt")) {
                const { convertToInput } = await import("../../extensions/core/widgetInputs.js");
                convertToInput(expand, expand.widgets.find(w => w.name === "prompt"));
            }
        } catch (error) { status.textContent = error.message; return; }
        nodes.forEach((node,i) => { app.graph.add(node); node.pos = positions[i]; });
        const connect = (from, outputName, to, inputName) => {
            const source = from.outputs?.findIndex(x => x.name === outputName);
            const target = to.inputs?.findIndex(x => x.name === inputName);
            if (source >= 0 && target >= 0) from.connect(source, to, target);
            else throw new Error(`Missing port ${outputName} → ${inputName}`);
        };
        try {
            connect(memory, "memory", recall, "memory");
            vision.connect(0, recall, recall.inputs.findIndex(x => x.name === "encoder"));
            connect(image, "IMAGE", recall, "image");
            connect(recall, "vrs_state", expand, "vrs_state");
            connect(recall, "request", expand, "prompt");
            connect(recall, "image", expand, "image");
            connect(expand, "tags", tags, "source");
            connect(expand, "natural_language", prose, "source");
            panel.close();
        } catch (error) { nodes.forEach(node => app.graph.remove(node)); status.textContent = error.message; }
        app.graph.setDirtyCanvas(true, true);
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
