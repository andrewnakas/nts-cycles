// nts/rt in a browser: runs the exported passes from interface.json alone (WebGPU), as export.Host does in Python.
// Files next to it: prep.wgsl, resolve.wgsl, interface.json (python -m nts.rt.export <dir>).
//
//   const host = await Host.create(device, "./", [H, W], [h, w], {depth: [1, 1 / near, 0], mvScale: [w, h]});
//   host.frame(encoder, colorTex, depthTex, motionTex, [ox, oy], reset);   // returns the texture holding the frame
//   device.queue.submit([encoder.finish()]);

export function evaluate(expr, names) {
  // + - * / ( ), numbers and names only: the interface's formulas, never code
  const tok = expr.match(/\d+\.?\d*(?:e[+-]?\d+)?|[A-Za-z_]\w*|[-+*/()]/g) || [];
  if (tok.join("") !== expr.replace(/\s+/g, "")) throw new Error(`bad formula: ${expr}`);
  let i = 0;
  const atom = () => {
    const t = tok[i++];
    if (t === "(") { const v = sum(); if (tok[i++] !== ")") throw new Error(`bad formula: ${expr}`); return v; }
    if (t === "-") return -atom();
    if (t === "+") return atom();
    if (/^\d/.test(t)) return parseFloat(t);
    if (t in names) return names[t];
    throw new Error(`unknown name ${t} in ${expr}`);
  };
  const prod = () => {
    let v = atom();
    while (tok[i] === "*" || tok[i] === "/") v = tok[i++] === "*" ? v * atom() : v / atom();
    return v;
  };
  const sum = () => {
    let v = prod();
    while (tok[i] === "+" || tok[i] === "-") v = tok[i++] === "+" ? v + prod() : v - prod();
    return v;
  };
  const v = sum();
  if (i !== tok.length) throw new Error(`bad formula: ${expr}`);
  return v;
}

export class Host {
  static async create(device, base, outHW, inHW, opts = {}) {
    const get = async (f) => { const r = await fetch(base + f); if (!r.ok) throw new Error(`${f}: ${r.status}`); return r; };
    const spec = await (await get("interface.json")).json();
    const src = {};
    for (const p of spec.passes) src[p.shader] = await (await get(p.shader)).text();
    return new Host(device, spec, src, outHW, inHW, opts);
  }

  constructor(device, spec, src, [H, W], [h, w], opts) {
    this.dev = device; this.spec = spec; this.W = W; this.H = H; this.w = w; this.h = h;
    const c = {};
    for (const [k, v] of Object.entries(spec.constants)) if (typeof v === "number") c[k] = v;
    Object.assign(c, opts.constants || {});
    // opts.constants replace the interface's fitted constants (foot, expand, wmax, shrink, prior, rel)
    const [mode, a, b] = opts.depth || [0, 0, 0];
    const [mx, my] = opts.mvScale || [1, 1];
    this.names = {...c, W, H, w, h, w_prev: w, h_prev: h, mv_x: mx, mv_y: my, depth_mode: mode, depth_a: a, depth_b: b};
    const U = GPUTextureUsage;
    const usage = {storage: U.STORAGE_BINDING, sampled: U.TEXTURE_BINDING, copy_src: U.COPY_SRC};
    const sizes = {render: [w, h], display: [W, H], "1x1": [1, 1]};
    this.res = {};
    for (const [name, r] of Object.entries(spec.resources)) {
      if (r.kind === "texture") {
        this.res[name] = device.createTexture({size: sizes[r.size], format: r.format,
                                               usage: r.usage.reduce((u, x) => u | usage[x], 0)});
      } else if (r.kind === "sampler") {
        this.res[name] = device.createSampler({magFilter: r.filter, minFilter: r.filter});
      } else if (r.kind === "uniform") {
        this.res[name] = device.createBuffer({size: r.bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
      }
    }
    this.modules = {};
    this.passes = spec.passes.map((p) => {
      const entries = p.bindings.map((bd) => {
        const e = {binding: bd.binding, visibility: GPUShaderStage.COMPUTE};
        if (bd.type === "uniform") e.buffer = {type: "uniform"};
        else if (bd.type === "texture") e.texture = {sampleType: bd.sample_type, viewDimension: "2d"};
        else if (bd.type === "storage") e.storageTexture = {access: bd.access, format: bd.format, viewDimension: "2d"};
        else e.sampler = {type: bd.sampler};
        return e;
      });
      const layout = device.createBindGroupLayout({entries});
      const module = this.modules[p.shader] = device.createShaderModule({code: src[p.shader], label: p.shader});
      const pipe = device.createComputePipeline({layout: device.createPipelineLayout({bindGroupLayouts: [layout]}),
                                                 compute: {module, entryPoint: spec.wgsl_entry}});
      return {p, layout, pipe};
    });
    this.t = 0;
  }

  params(offset, reset) {
    const n = {...this.names, ox: offset[0], oy: offset[1], reset: reset ? 1 : 0};
    const buf = new ArrayBuffer(this.spec.params.bytes);
    const dv = new DataView(buf);
    for (const f of this.spec.params.fields) {
      f.values.forEach((v, k) => {
        const x = evaluate(v, n);
        if (f.type === "vec4<u32>") dv.setUint32(f.offset + 4 * k, Math.trunc(x), true);
        else dv.setFloat32(f.offset + 4 * k, x, true);
      });
    }
    return buf;
  }

  frame(enc, color, depth, motion, offset, reset = false) {
    const cur = this.t % 2;
    const sub = {cur, prev: 1 - cur};
    const inputs = {color, depth, motion};
    this.dev.queue.writeBuffer(this.res.params, 0, this.params(offset, reset || this.t === 0));
    for (const {p, layout, pipe} of this.passes) {
      const entries = p.bindings.map((bd) => {
        const r = bd.resource.replace(/\{(\w+)\}/g, (_, k) => sub[k]);
        const obj = inputs[r] || this.res[r];
        if (bd.type === "uniform") return {binding: bd.binding, resource: {buffer: obj, offset: 0, size: this.spec.params.bytes}};
        if (bd.type === "sampler") return {binding: bd.binding, resource: obj};
        return {binding: bd.binding, resource: obj.createView()};
      });
      const group = this.dev.createBindGroup({layout, entries});
      const [gx, gy] = p.workgroup;
      const [w, h] = p.grid === "render" ? [this.w, this.h] : [this.W, this.H];
      const cp = enc.beginComputePass();
      cp.setPipeline(pipe);
      cp.setBindGroup(0, group);
      cp.dispatchWorkgroups(Math.ceil(w / gx), Math.ceil(h / gy));
      cp.end();
    }
    this.t += 1;
    return this.res[`hist${cur}`];
  }
}
