// GET /api/config -> public settings for the page. The Clerk publishable key is public by
// design (it names the Clerk instance); the secret key never leaves the server side.
export default function handler(req, res) {
  res.setHeader("cache-control", "s-maxage=300");
  return res.status(200).json({ clerkPublishableKey: process.env.CLERK_PUBLISHABLE_KEY || null });
}
