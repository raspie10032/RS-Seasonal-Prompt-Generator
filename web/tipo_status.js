import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const korean = (navigator.language || "").startsWith("ko");
const labels = {
    validating: ["입력 확인", "Validating input"],
    "preparing-models": ["모델 준비·다운로드", "Preparing / downloading models"],
    "generating-prose": ["장면 확장 중 (자연어)", "Expanding scene (prose)"],
    "generating-tags": ["태그 생성 중", "Generating tags"],
    complete: ["완료", "Complete"], empty: ["입력 없음 — 실행 생략", "No input — skipped"],
    failed: ["실패 — 실행 오류 확인", "Failed — check execution error"],
};

app.registerExtension({
    name: "RS.TIPO.Status",
    async setup() {
        api.addEventListener("rs_tipo_status", ({ detail }) => {
            const node = app.graph?.getNodeById(detail.node);
            if (!node?._rsStatus || !labels[detail.stage]) return;
            node._rsStatus.textContent = labels[detail.stage][korean ? 0 : 1];
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["GemmaTipoNative", "GemmaPromptExpandNative"].includes(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const text = document.createElement("div");
            text.style.cssText = "font-size:12px;padding:4px 8px;overflow-wrap:anywhere;";
            text.textContent = korean ? "대기 — both는 순차 생성 2회" : "Ready — both uses two sequential calls";
            this._rsStatus = text;
            this.addDOMWidget("rs_status", "rs_status", text, { serialize: false });
            return result;
        };
    },
});
