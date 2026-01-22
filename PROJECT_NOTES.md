# Silver Price Prediction System - Project Notes

## CRITICAL: READ THIS FILE BEFORE STARTING ANY WORK SESSION

---

## Server Information

| Item | Value |
|------|-------|
| **Server IP** | `62.72.58.74` |
| **SSH Access** | `ssh root@62.72.58.74` |
| **Hosting Provider** | Hostinger KVM8 (8-core, 32GB RAM, 400GB SSD) |

---

## Domains & Ports

| Service | Domain | Port | Internal URL |
|---------|--------|------|--------------|
| **Backend API** | `predictionapi.gahfaudio.in` | 8023 | `http://localhost:8023` |
| **Frontend** | `prediction.gahfaudio.in` | 8024 | `http://localhost:8024` |

---

## Git Repository

| Item | Value |
|------|-------|
| **Repository** | `https://github.com/rahul4webdev/silver-prediction.git` |
| **Server Path** | `/home/predictionapi.gahfaudio.in/public_html` |
| **Branch** | `main` |

---

## Database

| Item | Value |
|------|-------|
| **Type** | PostgreSQL |
| **Host** | `localhost` |
| **Port** | `5432` |
| **Database Name** | `silver_prediction` |
| **Username** | `prediction_user` |
| **Password** | `Luck@2028` (contains @, needs URL encoding: `Luck%402028`) |
| **Note** | TimescaleDB NOT installed (extension optional) |

---

## Services (systemd)

| Service Name | Description | Status Command |
|--------------|-------------|----------------|
| `silver-prediction-api` | FastAPI Backend (uvicorn) | `systemctl status silver-prediction-api` |
| `silver-prediction-tick-collector` | WebSocket Tick Collector | `systemctl status silver-prediction-tick-collector` |
| `silver-prediction-worker` | Celery Worker (if needed) | `systemctl status silver-prediction-worker` |

### Restart Commands:
```bash
systemctl restart silver-prediction-api
systemctl restart silver-prediction-tick-collector
```

### Log Commands:
```bash
journalctl -u silver-prediction-api -f
journalctl -u silver-prediction-tick-collector -f
tail -f /var/log/tick-collector.log
```

---

## Market Trading Hours (Indian Standard Time - IST)

| Market | Open | Close | Notes |
|--------|------|-------|-------|
| **MCX** | 9:00 AM | 11:55 PM | Main focus (via Upstox) |
| **COMEX** | 6:30 PM | 5:00 PM (next day) | Near 24h trading |

**Tick Collector Schedule**: Should run 9:00 AM to 11:55 PM IST daily

---

## API Keys & Authentication

| Service | Config Location | Notes |
|---------|-----------------|-------|
| **Upstox** | `.env` file | `UPSTOX_ACCESS_TOKEN` - needs daily refresh via OAuth |
| **Yahoo Finance** | N/A | Free, no API key needed |

---

## Data Sources

| Source | Market | Data Type | Method |
|--------|--------|-----------|--------|
| **Upstox WebSocket v3** | MCX | Real-time ticks | WebSocket (protobuf) |
| **Upstox REST API v2** | MCX | Historical candles | REST API |
| **Yahoo Finance** | COMEX | Historical candles | `yfinance` library |

---

## Project Structure

```
prediction/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # API endpoints
│   │   ├── core/             # Config, constants
│   │   ├── models/           # SQLAlchemy models
│   │   ├── services/         # Business logic
│   │   │   └── proto/        # Protobuf for Upstox
│   │   └── ml/               # ML models (to be implemented)
│   ├── workers/              # Background workers
│   └── main.py               # FastAPI entry point
├── frontend/                 # Next.js frontend
├── deploy/
│   └── systemd/              # Service files
└── PROJECT_NOTES.md          # THIS FILE
```

---

## What Has Been Implemented

### Phase 1: Foundation ✅
- [x] FastAPI backend with async support
- [x] PostgreSQL database with SQLAlchemy
- [x] Upstox OAuth2 authentication
- [x] Environment configuration with Pydantic

### Phase 2: Data Pipeline ✅
- [x] Upstox client for MCX historical data
- [x] Yahoo Finance client for COMEX data
- [x] Data sync service with upsert support
- [x] Historical data: MCX (~8 months), COMEX (~13 months)

### Phase 3: Real-Time Data Collection ✅
- [x] Tick data models (TickData, TickDataAggregated)
- [x] WebSocket tick collector (Upstox v3, protobuf)
- [x] Automatic tick aggregation (1s, 5s, 10s, 1m candles)
- [x] Tick stats API endpoints
- [x] systemd service for continuous collection

### Phase 4: ML Models ✅
- [x] Feature engineering (technical indicators) - RSI, MACD, Bollinger, ATR, etc.
- [x] Prophet model for trend/seasonality
- [x] LSTM model for sequence patterns (with Monte Carlo dropout for uncertainty)
- [x] XGBoost for feature importance
- [x] Ensemble voting mechanism with dynamic weight adjustment

### Phase 5: Prediction & Accuracy ✅
- [x] Prediction generation scheduler
- [x] Confidence interval calculations (50%, 80%, 95%)
- [x] Prediction verification worker (auto-verifies when target_time passes)
- [x] Accuracy tracking dashboard (frontend and API)
- [x] Prediction database model with full tracking

### Phase 6: News Sentiment Analysis (PENDING)
- [ ] News data collection (sources TBD)
- [ ] Sentiment analysis pipeline
- [ ] Historical event correlation
- [ ] Integration with prediction model

---

## API Endpoints

### Health & Auth
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/upstox` - Upstox connection status
- `GET /api/v1/auth/upstox/login` - OAuth login URL
- `GET /api/v1/auth/upstox/callback` - OAuth callback

### Historical Data
- `GET /api/v1/historical/{asset}/{market}` - Get historical candles
- `POST /api/v1/historical/sync-all` - Sync all data sources

### Tick Data
- `GET /api/v1/ticks/stats` - Tick collection statistics
- `GET /api/v1/ticks/recent` - Recent tick data
- `GET /api/v1/ticks/aggregated` - Aggregated candles

### Predictions
- `GET /api/v1/predictions/latest` - Latest prediction
- `POST /api/v1/predictions/generate` - Generate prediction
- `GET /api/v1/predictions/history` - Prediction history with filters
- `POST /api/v1/predictions/verify` - Manually trigger verification
- `POST /api/v1/predictions/train` - Train ML models

### Accuracy
- `GET /api/v1/accuracy/summary` - Accuracy metrics summary

---

## Known Issues & Fixes

### 1. Database Password with Special Characters
**Problem**: Password `Luck@2028` contains `@` which breaks the connection string.
**Fix**: URL encode the password in `config.py`:
```python
from urllib.parse import quote_plus
encoded_password = quote_plus(self.postgres_password)
```

### 2. TimescaleDB Not Available
**Problem**: TimescaleDB extension not installed on server.
**Fix**: Made it optional - tables work without it.

### 3. Timezone-Aware Datetime
**Problem**: Comparing naive and aware datetimes.
**Fix**: Always use `datetime.now(timezone.utc)` instead of `datetime.now()`.

### 4. Upstox v2 → v3 WebSocket
**Problem**: v2 endpoint discontinued (HTTP 410).
**Fix**: Updated to v3 endpoint: `https://api.upstox.com/v3/feed/market-data-feed/authorize`

### 5. Protobuf Version Compatibility
**Problem**: Generated protobuf code requires newer protobuf library.
**Fix**: Regenerate on server with `grpcio-tools==1.59.0` and `protobuf==4.25.0`.

---

## Deployment Checklist

After pushing changes to GitHub:

```bash
# SSH to server
ssh root@62.72.58.74

# Pull latest code
cd /home/predictionapi.gahfaudio.in/public_html
git pull origin main

# Restart services as needed
systemctl restart silver-prediction-api
systemctl restart silver-prediction-tick-collector

# Check logs
journalctl -u silver-prediction-api -n 50
```

---

## Future Enhancements

1. **News Sentiment Analysis**
   - Integrate financial news APIs
   - Build sentiment scoring model
   - Correlate historical events with price movements

2. **Model Improvements**
   - Add more correlated factors
   - Implement walk-forward validation
   - Dynamic model weight adjustment

3. **Alerting System**
   - Email/SMS alerts for predictions
   - Price threshold notifications

4. **MCX-COMEX Arbitrage**
   - Real-time spread monitoring
   - Arbitrage opportunity detection

---

## Contact & Resources

- **Upstox API Docs**: https://upstox.com/developer/api-documentation/
- **Upstox Community**: https://community.upstox.com/
- **Yahoo Finance**: https://pypi.org/project/yfinance/

---

*Last Updated: 2026-01-22*
