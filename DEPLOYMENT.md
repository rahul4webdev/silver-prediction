# Silver Price Prediction System - Deployment Guide

## Server Configuration

- **OS**: AlmaLinux 9
- **Control Panel**: CyberPanel with OpenLiteSpeed
- **API Domain**: predictionapi.gahfaudio.in
- **Frontend Domain**: prediction.gahfaudio.in
- **Repository**: https://github.com/rahul4webdev/silver-prediction.git

## Prerequisites (Already Installed)

- Python 3.11
- PostgreSQL
- Node.js
- Redis

---

## Step 1: Clone Repository to Both Sites

### Backend (API):
```bash
cd /home/predictionapi.gahfaudio.in/public_html
rm -rf * .* 2>/dev/null || true
git clone https://github.com/rahul4webdev/silver-prediction.git .
```

### Frontend:
```bash
cd /home/prediction.gahfaudio.in/public_html
rm -rf * .* 2>/dev/null || true
git clone https://github.com/rahul4webdev/silver-prediction.git .
```

---

## Step 2: Setup Backend

```bash
cd /home/predictionapi.gahfaudio.in/public_html

# Create data directories
mkdir -p /home/predictionapi.gahfaudio.in/data/{models,raw,processed}
mkdir -p /home/predictionapi.gahfaudio.in/logs

# Create virtual environment
python3.11 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

## Step 3: Setup Frontend

```bash
cd /home/prediction.gahfaudio.in/public_html/frontend

# Install dependencies
npm ci

# Create .env.local
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=https://predictionapi.gahfaudio.in
NEXT_PUBLIC_WS_URL=wss://predictionapi.gahfaudio.in
NEXT_PUBLIC_APP_NAME=Silver Prediction
EOF

# Build
npm run build

# Start with PM2 on port 8024
pm2 start npm --name "silver-prediction-frontend" -- start -- -p 8024
pm2 save
pm2 startup
```

---

## Step 4: Configure Supervisor for Backend Services

```bash
# Copy supervisor configs
sudo cp /home/predictionapi.gahfaudio.in/public_html/deploy/supervisor/*.conf /etc/supervisord.d/

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Start services
sudo supervisorctl start silver-prediction-api
sudo supervisorctl start silver-prediction-worker
sudo supervisorctl start silver-prediction-scheduler

# Check status
sudo supervisorctl status
```

---

## Step 5: Configure OpenLiteSpeed vHost

### For API (predictionapi.gahfaudio.in):

Go to **CyberPanel → Websites → predictionapi.gahfaudio.in → vHost Conf**

**ADD these sections to your existing config** (after the existing `extprocessor predi3169` section):

```apache
# FastAPI Backend Proxy
extprocessor fastapi_backend {
  type                    proxy
  address                 127.0.0.1:8023
  maxConns                100
  pcKeepAliveTimeout      60
  initTimeout             60
  retryTimeout            0
  respBuffer              0
}

# WebSocket Proxy
extprocessor websocket_backend {
  type                    proxy
  address                 127.0.0.1:8023
  maxConns                100
  pcKeepAliveTimeout      300
  initTimeout             60
  retryTimeout            0
  respBuffer              0
}

# Proxy all API requests to FastAPI
context / {
  type                    proxy
  handler                 fastapi_backend
  addDefaultCharset       off
}

# WebSocket context
context /ws {
  type                    proxy
  handler                 websocket_backend
  addDefaultCharset       off
}
```

**REPLACE your existing `rewrite` block with:**

```apache
rewrite  {
  enable                  1
  autoLoadHtaccess        1
  rules                   <<<END_RULES
RewriteEngine On
RewriteCond %{HTTP:Upgrade} websocket [NC]
RewriteCond %{HTTP:Connection} upgrade [NC]
RewriteRule ^/ws/(.*)$ ws://127.0.0.1:8023/ws/$1 [P,L]
RewriteRule ^(.*)$ http://127.0.0.1:8023$1 [P,L]
END_RULES
}
```

---

### For Frontend (prediction.gahfaudio.in):

Go to **CyberPanel → Websites → prediction.gahfaudio.in → vHost Conf**

**ADD these sections to your existing config** (after the existing `extprocessor predi8898` section):

```apache
# Next.js Frontend Proxy
extprocessor nextjs_frontend {
  type                    proxy
  address                 127.0.0.1:8024
  maxConns                100
  pcKeepAliveTimeout      60
  initTimeout             60
  retryTimeout            0
  respBuffer              0
}

# Proxy all requests to Next.js
context / {
  type                    proxy
  handler                 nextjs_frontend
  addDefaultCharset       off
}

# Static assets with caching
context /_next/static {
  type                    proxy
  handler                 nextjs_frontend
  addDefaultCharset       off
  extraHeaders            <<<END_HEADERS
Cache-Control: public, max-age=31536000, immutable
END_HEADERS
}
```

**REPLACE your existing `rewrite` block with:**

```apache
rewrite  {
  enable                  1
  autoLoadHtaccess        1
  rules                   <<<END_RULES
RewriteEngine On
RewriteRule ^(.*)$ http://127.0.0.1:8024$1 [P,L]
END_RULES
}
```

---

## Step 6: Restart OpenLiteSpeed

```bash
sudo systemctl restart lsws
```

---

## Step 7: Verify Installation

```bash
# Check backend services
sudo supervisorctl status

# Check frontend
pm2 status

# Test API health
curl https://predictionapi.gahfaudio.in/api/v1/health

# Check backend logs
tail -f /home/predictionapi.gahfaudio.in/logs/api.log

# Check frontend logs
pm2 logs silver-prediction-frontend
```

---

## GitHub Actions Secrets (Already Configured)

| Secret | Description |
|--------|-------------|
| `SERVER_HOST` | Your server IP |
| `SERVER_USER` | SSH username |
| `SERVER_SSH_KEY` | SSH private key |
| `SERVER_SSH_PORT` | SSH port |
| `UPSTOX_API_KEY` | Upstox API key |
| `UPSTOX_API_SECRET` | Upstox secret |
| `UPSTOX_ACCESS_TOKEN` | Upstox access token |
| `SECRET_KEY` | App secret key |
| `POSTGRES_HOST` | Database host |
| `POSTGRES_PORT` | Database port |
| `POSTGRES_DB` | Database name |
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `REDIS_HOST` | Redis host |
| `REDIS_PORT` | Redis port |

---

## Troubleshooting

### API not responding
```bash
sudo supervisorctl status silver-prediction-api
tail -f /home/predictionapi.gahfaudio.in/logs/api.log
sudo supervisorctl restart silver-prediction-api
```

### Frontend not loading
```bash
pm2 status
pm2 logs silver-prediction-frontend
pm2 restart silver-prediction-frontend
```

### 503 Service Unavailable
- Check if FastAPI is running on port 8023
- Check if Next.js is running on port 8024
- Verify the extprocessor configs are correct

### WebSocket not connecting
- Ensure the WebSocket rewrite rules are in place
- Check if the backend supports WebSocket on `/ws/` path

---

## File Locations

| Component | Path |
|-----------|------|
| Backend code | `/home/predictionapi.gahfaudio.in/public_html/backend` |
| Backend venv | `/home/predictionapi.gahfaudio.in/public_html/venv` |
| Backend .env | `/home/predictionapi.gahfaudio.in/public_html/.env` |
| Backend logs | `/home/predictionapi.gahfaudio.in/logs/` |
| Model data | `/home/predictionapi.gahfaudio.in/data/` |
| Frontend code | `/home/prediction.gahfaudio.in/public_html/frontend` |
| Supervisor configs | `/etc/supervisord.d/` |
