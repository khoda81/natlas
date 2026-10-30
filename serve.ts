// Development server: the page, bundled on request, and the datasets in public/.

import index from "./index.html";
import playground from "./playground.html";

const server = Bun.serve({
  hostname: "localhost",
  port: Number(process.env.PORT ?? 5173),
  development: true,
  routes: { "/": index, "/playground": playground, "/playground.html": playground },
  async fetch(request) {
    const path = new URL(request.url).pathname;
    if (path === "/playground-worker.js") {
      const build = await Bun.build({ entrypoints: ["./src/playground/worker.ts"], target: "browser" });
      return build.success ? new Response(build.outputs[0], { headers: { "Content-Type": "text/javascript" } })
        : new Response(build.logs.join("\n"), { status: 500 });
    }
    if (path === "/partition.proto") return new Response(Bun.file("proto/natlas/partition/v1/partition.proto"), { headers: { "Content-Type": "text/plain" } });
    const file = Bun.file(`public${path}`);
    return (await file.exists()) ? new Response(file) : new Response("not found", { status: 404 });
  },
});

console.log(`natlas at ${server.url}`);
