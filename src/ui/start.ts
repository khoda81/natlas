import { App } from "./main";

void new App().start().catch((error: unknown) => {
  const status = document.getElementById("status")!;
  status.textContent = String(error);
  status.hidden = false;
});
