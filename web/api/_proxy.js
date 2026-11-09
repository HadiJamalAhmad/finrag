// Forwards a question to the finrag backend on Modal, adding the API key server-side.
// Files starting with "_" are not deployed as routes by Vercel.
//
// Env (Vercel project settings): FINRAG_BACKEND_URL, FINRAG_API_KEY.

export async function forward(req, res, path) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ detail: "POST only" });
  }
  const question = typeof req.body?.question === "string" ? req.body.question.trim() : "";
  if (question.length < 3 || question.length > 500) {
    return res.status(400).json({ detail: "question must be 3 to 500 characters" });
  }
  // A preset name (llama-3b | qwen3-4b | mistral-24b); the backend validates it (422).
  const model = typeof req.body?.model === "string" ? req.body.model : undefined;
  if (model !== undefined && !/^[a-z0-9.-]{1,40}$/.test(model)) {
    return res.status(400).json({ detail: "invalid model name" });
  }
  // Identity (SIMULATED for the demo; in production the SSO gateway sets the groups and
  // the browser never can): the page only picks a profile, the groups are decided here.
  const PROFILES = { public: undefined, agence: ["agence"], risques: ["risques"] };
  const profile = typeof req.body?.profile === "string" ? req.body.profile : "public";
  if (!(profile in PROFILES)) return res.status(400).json({ detail: "unknown profile" });
  const groups = PROFILES[profile];
  const base = process.env.FINRAG_BACKEND_URL;
  const key = process.env.FINRAG_API_KEY;
  if (!base || !key) {
    return res.status(500).json({ detail: "backend not configured" });
  }
  try {
    const upstream = await fetch(`${base.replace(/\/$/, "")}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key },
      body: JSON.stringify({ question, ...(model ? { model } : {}), ...(groups ? { groups } : {}) }),
      signal: AbortSignal.timeout(290_000),
    });
    const text = await upstream.text();
    res.status(upstream.status).setHeader("content-type", "application/json");
    return res.send(text);
  } catch (err) {
    return res.status(504).json({ detail: `backend unreachable: ${err.name}` });
  }
}
