// /api/graphql -> backend /graphql (Strawberry), with the API key added server-side.
//   GET  (a browser): the GraphiQL explorer. It posts back to this same URL.
//   POST (a query):   forwarded as is; size-capped. Queries never go over GET.
// Same guards as the backend: the answer path, refusals, costs and traces are REST's.

const MAX_BODY = 20_000; // characters: a query, never a payload

export default async function handler(req, res) {
  const base = process.env.FINRAG_BACKEND_URL;
  const key = process.env.FINRAG_API_KEY;
  if (!base || !key) return res.status(500).json({ detail: "backend not configured" });
  const url = `${base.replace(/\/$/, "")}/graphql`;

  try {
    if (req.method === "GET") {
      const upstream = await fetch(url, {
        headers: { "x-api-key": key, accept: "text/html" },
        signal: AbortSignal.timeout(290_000),
      });
      res.status(upstream.status).setHeader("content-type", "text/html; charset=utf-8");
      return res.send(await upstream.text());
    }
    if (req.method !== "POST") {
      res.setHeader("Allow", "GET, POST");
      return res.status(405).json({ detail: "GET (explorer) or POST (query) only" });
    }
    const body = typeof req.body === "string" ? req.body : JSON.stringify(req.body ?? {});
    if (body.length > MAX_BODY) return res.status(413).json({ detail: "query too large" });
    const upstream = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key },
      body,
      signal: AbortSignal.timeout(290_000),
    });
    res.status(upstream.status).setHeader("content-type", "application/json");
    return res.send(await upstream.text());
  } catch (err) {
    return res.status(504).json({ detail: `backend unreachable: ${err.name}` });
  }
}
