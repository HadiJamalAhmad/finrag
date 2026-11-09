// POST /api/agent -> backend /agent (tool-calling agent; response includes the audit log)
import { forward } from "./_proxy.js";

export default function handler(req, res) {
  return forward(req, res, "/agent");
}
