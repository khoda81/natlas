// Run `bun scripts/bench-frames.ts`, open the printed URL, and click Run.
// NATLAS_PERF_ROOT can point at a baseline checkout exporting App from main.ts.
import { resolve } from "node:path";

const root = resolve(process.env.NATLAS_PERF_ROOT ?? ".");
const out = resolve(process.env.NATLAS_PERF_OUTPUT ?? "/tmp/natlas-frame-benchmark");
const html = (await Bun.file("index.html").text())
  .replace('<script type="module" src="./src/ui/start.ts"></script>', '<script type="module" src="/benchmark.js"></script>')
  .replace("</body>", '<aside style="position:fixed;bottom:25px;right:8px;z-index:10;background:#fff;color:#111;padding:8px"><button id="benchmark-run">Run frame benchmark</button><pre id="benchmark-output">Ready</pre></aside></body>');
const built = await Bun.build({ entrypoints: ["tests/frame-benchmark.ts"], target: "browser", minify: true,
  plugins: [{ name: "benchmark-checkout", setup(build) {
    build.onResolve({ filter: /^\.\.\/src\// }, args => ({ path: `${resolve(root, "tests", args.path)}.ts` }));
    build.onLoad({ filter: /\/src\/ui\/main\.ts$/ }, async args => {
      const source = await Bun.file(args.path).text();
      // Older checkouts combined startup and App. Export the same class for
      // the harness, omitting only the unrelated network/bootstrap call.
      return { contents: source.replace(/^class App \{/m, "export class App {")
        .replace(/void new App\(\)\.start\(\)\.catch\([\s\S]*$/, ""), loader: "ts" };
    });
  } }] });
if (!built.success) throw new Error(String(built.logs));
const bundle = await built.outputs[0]!.text();
const headers = { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" };
const server = Bun.serve({ hostname: "127.0.0.1", port: Number(process.env.PORT ?? 5187),
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/result" && request.method === "POST") {
      const result = await request.json() as { label: string };
      if (!/^[a-z0-9-]+$/.test(result.label)) return new Response("invalid label", { status: 400 });
      await Bun.write(`${out}/${result.label}.json`, JSON.stringify(result, null, 2));
      return new Response("saved", { headers });
    }
    if (url.pathname === "/") return new Response(html, { headers: { ...headers, "content-type": "text/html" } });
    if (url.pathname === "/benchmark.js") return new Response(bundle, { headers: { ...headers, "content-type": "text/javascript" } });
    if (url.pathname === "/fixture" && process.env.NATLAS_PERF_FIXTURE) return new Response(Bun.file(process.env.NATLAS_PERF_FIXTURE), { headers });
    if (url.pathname === "/generated/natlas_core_bg.wasm") return new Response(Bun.file(`${root}/src/core/generated/natlas_core_bg.wasm`), { headers: { ...headers, "content-type": "application/wasm" } });
    if (url.pathname === "/src/ui/style.css") return new Response(Bun.file(`${root}/src/ui/style.css`), { headers });
    return new Response("not found", { status: 404 });
  },
});
console.log(`Frame benchmark: ${server.url} (reports: ${out})`);
