// POST /api/query -> backend /query (scope -> retrieval -> evidence gate -> LLM -> numbers)
import { forward } from "./_proxy.js";

export default function handler(req, res) {
  return forward(req, res, "/query");
}
