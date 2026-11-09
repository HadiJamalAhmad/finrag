// GET /api/me -> backend /me: who the backend thinks is asking (from the verified Clerk
// token) and which groups that unlocks. The token is forwarded untouched.
export default async function handler(req, res) {
  const base = process.env.FINRAG_BACKEND_URL;
  const key = process.env.FINRAG_API_KEY;
  if (!base || !key) return res.status(500).json({ detail: "backend not configured" });
  const headers = { "x-api-key": key };
  if (req.headers.authorization) headers.authorization = req.headers.authorization;
  try {
    const upstream = await fetch(`${base.replace(/\/$/, "")}/me`, {
      headers, signal: AbortSignal.timeout(290_000),
    });
    res.status(upstream.status).setHeader("content-type", "application/json");
    res.setHeader("cache-control", "no-store");
    return res.send(await upstream.text());
  } catch (err) {
    return res.status(504).json({ detail: `backend unreachable: ${err.name}` });
  }
}
