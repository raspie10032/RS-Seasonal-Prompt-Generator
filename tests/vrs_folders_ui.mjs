// Exercise the shipped UI logic with a local DOM/API harness; no browser or network.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.checked = false; this.disabled = false; this.textContent = ""; this.listeners = {}; }
    append(node) { node.parent = this; this.children.push(node); }
    get options() { return this.children.filter(n => n.tag === "option"); }
    get value() { return this.tag === "select" ? (this.options.find(o => o.value === this._value) || this.options[0])?.value ?? "" : this._value ?? ""; }
    set value(value) { this._value = String(value); }
    replaceChildren() { this.children = []; this._value = ""; }
    setAttribute(name, value) { this[name] = value; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
    focus() {}
    remove() { this.parent.children = this.parent.children.filter(n => n !== this); }
}
const config = { id: "a".repeat(32), label: "Existing", roots: ["/old"], batch_size: 4,
    model_name: "(auto: Gemma FP16 vision)", tagger: "none", use_cpu: true };
const catalog = (saved = 0) => ({config, snapshot: saved ? "b".repeat(64) : null, search_snapshot: saved ? "b".repeat(64) : null,
    records: {encoded:saved}, search_count: saved, files: {}, errors: [], tagging: {}, store_path:"/memory"});
const status = (c = null) => ({collections: c ? [config] : [], collection:c, active:false, state:"stopped", models:[config.model_name], output_directory:"/output"});
let current = status();
let missing = [];
let timer;
let startError;
const calls = [];
const body = new Element("body");
const app = {registerExtension(value) { this.extension = value; }};
const api = { async fetchApi(url, options = {}) {
    const action = url.split("/").at(-1).split("?")[0];
    const input = options.body ? JSON.parse(options.body) : null;
    calls.push({action, input});
    let data = structuredClone(current);
    if (url.includes("?collection_id=")) data.selected_collection = data.collection;
    if (action === "preview") data = {can_start:!missing.length, missing_packages:missing,
        samples:[{path:"/output",images_checked:2,readable_metadata:1,no_metadata:1,read_errors:0,limited:true}], automatic_vision:true, use_cpu:!!input.use_cpu};
    if (action === "start") {
        if (startError) return {ok:false,json:async () => ({error:startError})};
        current = status(catalog());
        current.active = true;
        current.collection.config = {...config, use_cpu:input.use_cpu ?? false};
        current.collections = [current.collection.config];
        data = structuredClone(current);
    }
    return {ok:true,json:async () => data};
}};
const source = fs.readFileSync(new URL("../web/vrs_folders.js", import.meta.url), "utf8").replace(/^import .*;\n/gm, "");
vm.runInNewContext(source, {app, api, navigator:{language:"en"}, document:{body,createElement:tag => new Element(tag)},
    setTimeout: callback => { timer = callback; return 1; }, clearTimeout: () => { timer = null; }, LiteGraph:{}, console});
await app.extension.setup();
await body.children[0].onclick();
let panel = body.children.at(-1);
const buttons = () => panel.children.find(n => n.tag === "div").children;
const button = text => buttons().find(n => n.textContent === text);
const fields = () => panel.children.filter(n => n.tag === "label").map(n => n.children[0]);
const [select, label, roots, output, model, batch, tagger, cpu] = fields();
assert.equal(button("Add memory node").disabled, true, "empty collection has no usable memory");
assert.equal(button("Add VRS → TIPO workflow").disabled, true);
label.value = "My draft"; roots.value = "/new"; batch.value = 3;
await button("Check folders & setup").onclick();
assert.equal(calls.at(-1).action, "preview");
assert.equal(calls.at(-1).input.use_cpu, false);
assert.equal(calls.filter(c => c.action === "start").length, 0, "check must not start");
assert.match(panel.children.find(n => n.tag === "pre").textContent, /partial sample/);

// Existing collection -> fresh draft must restore paths, name and explicit CPU choice.
current = status(catalog());
await timer();
select.value = config.id;
await select.onchange();
assert.equal(roots.value, "/old");
assert.equal(cpu.checked, true);
assert.equal(button("Add memory node").disabled, true, "saved config without experiences is not ready");
select.value = "";
await select.onchange();
assert.equal(roots.value, "/new");
assert.equal(label.value, "My draft");
assert.equal(batch.value, "3");
assert.equal(cpu.checked, false, "existing CPU consent must not leak into a new collection");

missing = ["timm"];
tagger.value = "pixai";
await button("Start / Resume").onclick();
assert.equal(calls.filter(c => c.action === "start").length, 0, "missing optional deps block submission");
assert.match(panel.children.find(n => n.tag === "pre").textContent, /requirements-tagger.txt/);
missing = [];
tagger.value = "none";
startError = "Device setting changed; create a separate collection";
await button("Start / Resume").onclick();
assert.match(panel.children.find(n => n.tag === "pre").textContent, /Device setting changed/);
await timer();
assert.match(panel.children.find(n => n.tag === "pre").textContent, /Device setting changed/, "poll must not erase action errors");
startError = null;
await button("Start / Resume").onclick();
assert.equal(select.value, config.id, "created collection selected immediately, without a stale new form");
assert.equal(roots.disabled, true);
assert.equal(button("Add memory node").disabled, true);
assert.ok(panel.children.some(n => n.tag === "p" && n.textContent.includes("Collection is running")));
current = status(catalog(5));
await timer();
assert.equal(button("Add memory node").disabled, false);
assert.equal(button("Add VRS → TIPO workflow").disabled, false);
assert.match(panel.children.find(n => n.tag === "p" && n.textContent.includes("Ready to search")).textContent, /Load Image/);
const progress = panel.children.find(n => n.tag === "progress");
assert.equal(Number(progress.value), 5);
assert.equal(progress.max, 5);
panel.close();
assert.equal(timer, null, "close stops polling");
console.log("VRS folders UI: setup, draft/consent isolation, missing deps, action errors, created selection, ready gate and close passed");
