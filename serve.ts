// Development server: the page, bundled on request, and the datasets in public/.

import index from "./index.html";

const server = Bun.serve({
  hostname: "localhost",
  port: Number(process.env.PORT ?? 5173),
  development: true,
  routes: { "/": index },
  async fetch(request) {
    const file = Bun.file(`public${new URL(request.url).pathname}`);
    return (await file.exists()) ? new Response(file) : new Response("not found", { status: 404 });
  },
});

console.log(`natlas at ${server.url}`);
