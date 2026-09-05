import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app.js";
import "./global.css";

const root = document.getElementById("root");
if (root === null) throw new Error("SwarmX renderer root is missing.");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
