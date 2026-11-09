// GET /api/models -> backend /models: which presets exist and which are available
// (a hosted model without its key is listed but unavailable).
export default async function handler(req, res) {
  const base = process.env.FINRAG_BACKEND_URL;
  const key = process.env.FINRAG_API_KEY;
  if (!base || !key) return res.status(500).json({ detail: "backend not configured" });
  try {
    const upstream = await fetch(`${base.replace(/\/$/, "")}/models`, {
      headers: { "x-api-key": key },
      signal: AbortSignal.timeout(290_000),
    });
    res.status(upstream.status).setHeader("content-type", "application/json");
    res.setHeader("cache-control", "s-maxage=60");
    return res.send(await upstream.text());
  } catch (err) {
    return res.status(504).json({ detail: `backend unreachable: ${err.name}` });
  }
}
