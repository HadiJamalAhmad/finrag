# Free fallback host: Oracle Cloud Always Free

While the Modal workspace is off, the backend can run on an Oracle Always Free VM
(Ampere A1, arm64). Modal stays the main target: `deploy/modal_app.py` is unchanged, and
going back is two settings (see [Back to Modal](#back-to-modal)).

| | Modal (main) | Oracle (fallback) |
|---|---|---|
| Hardware | L4 / A10G GPU, scale to zero | 2 Ampere cores, 12 GB RAM, always on |
| Open models | llama-3b, qwen3-4b, mistral-24b | llama-3b only (the others: 503, "not configured" on the page) |
| Hosted models | GPT, Claude (if keys are set) | same |
| Answer time | seconds once warm | measured after the first deploy (CPU: expect tens of seconds) |
| Cost shown | GPU seconds x price | 0 (`FINRAG_GPU=A1-CPU`) |

Same API, same `x-api-key` rule (`server.py`), same pinned digest (`pull_models.py` stops
if Ollama serves other weights), same index (data/ copied from the PC).

## 1. Create the VM (once)

1. Create an Oracle Cloud account (a card is asked for verification; Always Free resources
   are not billed). Pick an EU home region: it cannot be changed later.
2. Compute → Instances → Create instance:
   - Image: **Canonical Ubuntu 24.04** (aarch64).
   - Shape: **Ampere VM.Standard.A1.Flex**, **2 OCPU, 12 GB** (the Always Free allowance is
     4 OCPU / 24 GB in total: the other half is for JobHack).
   - Networking: assign a public IPv4 address.
   - SSH keys: download the private key (keep it out of every repository).
   - "Out of capacity": retry later or in another availability domain.
3. Networking → the VM's subnet → Security list → add ingress rules for TCP **80** and
   **443** from `0.0.0.0/0`.

## 2. Prepare the VM

```bash
ssh -i <private-key> ubuntu@<public-ip>

# Oracle's Ubuntu image also blocks ports in iptables: open 80 and 443
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && exit     # log in again for the group to apply
```

## 3. Code, data, secrets

```bash
# on the VM
git clone https://github.com/HadiJamalAhmad/finrag.git && mkdir -p finrag/data
```

```bash
# on the PC, from the repo root: the corpus (gitignored, never in git)
scp -i <private-key> -r data/qdrant data/facts data/manifest.json data/company_tickers.json \
    ubuntu@<public-ip>:~/finrag/data/
```

```bash
# on the VM
cd finrag
cp deploy/oracle/.env.example deploy/oracle/.env
nano deploy/oracle/.env
```

In `.env`: `DOMAIN=<public-ip-with-dashes>.sslip.io` (for 141.145.12.34:
`141-145-12-34.sslip.io`, a free name that resolves to that IP, so Caddy can get a Let's
Encrypt certificate), then the same `FINRAG_API_KEY`, `FINRAG_INGEST_KEY`, Clerk and
Langfuse values as on Modal. Type them in the file on the VM; never paste a key in a chat.

## 4. Start

```bash
docker compose -f deploy/oracle/compose.yml up -d --build
docker compose -f deploy/oracle/compose.yml logs -f models   # pull + digest check + load
curl https://<DOMAIN>/health                                  # {"status":"ok"}
```

The first start pulls Llama 3.2 3B (about 2 GB). The `models` container must end with
`digests match` and `llama-3b loaded`; the API starts only then.

## 5. Point the site at it

- Vercel → project finrag → Settings → Environment Variables:
  `FINRAG_BACKEND_URL = https://<DOMAIN>`, then redeploy.
- GitHub → finrag → Settings → Secrets and variables → Actions → **Variables**:
  `FINRAG_DEPLOY_TARGET = oracle` (CI then skips the Modal deploy instead of failing).

Update after a merge to main: `git pull && docker compose -f deploy/oracle/compose.yml up -d --build`.
After a new `finrag ingest`: copy data/ again (step 3), then `docker compose ... restart api`.

## Back to Modal

1. Vercel: `FINRAG_BACKEND_URL` = the Modal URL again, redeploy.
2. GitHub: delete the variable `FINRAG_DEPLOY_TARGET`; the next push to main deploys to Modal.
3. The VM can stay (free) as a spare, or be stopped: `docker compose -f deploy/oracle/compose.yml down`.
