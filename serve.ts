// Development server: the page, bundled on request, and the datasets in public/.

import index from "./index.html";

const server = Bun.serve({
  hostname: "localhost",
  port: Number(process.env.PORT ?? 5173),
  development: true,
  routes: { "/": index },
  async fetch(request) {
    const { pathname } = new URL(request.url);
    if (pathname === "/generated/natlas_core_bg.wasm") {
      return new Response(Bun.file(`src/core/generated${pathname.slice(10)}`), {
        headers: { "content-type": "application/wasm" },
      });
    }
    const file = Bun.file(`public${pathname}`);
    return (await file.exists()) ? new Response(file) : new Response("not found", { status: 404 });
  },
});

console.log(`natlas at ${server.url}`);
