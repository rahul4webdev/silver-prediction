# Silver Price Prediction System - Architecture Documentation

## Last Updated: 2026-01-27

---

## 1. PREDICTION REQUIREMENTS

### MCX Silver Predictions
- **Contracts**: 3 types (SILVER, SILVERM, SILVERMIC) - each with nearest expiry
- **Intervals**: 30m, 1h, 4h, 1d
- **Schedule**: Only during market hours (9:00 AM - 11:30 PM IST, Mon-Fri)

| Interval | Frequency | Predictions/Cycle | Daily Total |
|----------|-----------|-------------------|-------------|
| 30m | Every 30 min | 3 | 84 (28 cycles × 3) |
| 1h | Every hour | 3 | 42 (14 cycles × 3) |
| 4h | Every 4 hours | 3 | 12 (4 cycles × 3) |
| 1d | Once daily | 3 | 3 |
| **Total MCX** | | | **141** |

### COMEX Silver Predictions
- **Contracts**: Single (SI=F futures)
- **Intervals**: 30m, 1h, 4h, 1d
- **Schedule**: Nearly 24/5 (closed weekends)

| Interval | Frequency | Daily Total |
|----------|-----------|-------------|
| 30m | Every 30 min | 28 |
| 1h | Every hour | 14 |
| 4h | Every 4 hours | 4 |
| 1d | Once daily | 1 |
| **Total COMEX** | | **47** |

---

## 2. VERIFICATION REQUIREMENTS

### Timing
- 30m predictions: Verify at target_time (30 min after prediction)
- 1h predictions: Verify at target_time (1 hour after prediction)
- 4h predictions: Verify at target_time (4 hours after prediction)
- 1d predictions: Verify at target_time (next day)

### Actual Price Source
- **MCX**: Upstox API live quote OR tick_data OR price_data
- **COMEX**: Yahoo Finance OR price_data

### Verification Process
1. Find predictions where `target_time < now` AND `verified_at IS NULL`
2. Get actual price at `target_time` from Upstox/Yahoo/tick_data
3. Compare predicted direction vs actual direction
4. Calculate error percentage and CI coverage
5. Update prediction record with results

---

## 3. SCHEDULER ARCHITECTURE

### IMPORTANT: Two Schedulers Running (DUPLICATION ISSUE)

#### APScheduler (`backend/workers/scheduler.py`)
- **Service**: `silver-prediction-scheduler.service`
- **Framework**: AsyncIOScheduler
- **Responsible for**:
  - Data sync (every 30 min)
  - Model training (daily at 6 AM IST + hourly during market)
  - Prediction generation (all intervals)
  - Prediction verification (every 5 min)

#### Celery Beat (`backend/workers/prediction_verifier.py`)
- **Service**: `silver-prediction-beat.service` + `silver-prediction-worker.service`
- **Framework**: Celery + Celery Beat
- **Responsible for**: Same tasks as APScheduler (DUPLICATE!)

### Current Issue
Both schedulers try to do the same work, causing:
- Duplicate predictions
- Race conditions in verification
- Inconsistent state

### Resolution
**Use only APScheduler** - it's more tightly integrated with the codebase.
Disable or repurpose Celery for other async tasks.

---

## 4. DATA FLOW

### Price Data Collection
```
Upstox API (MCX) ──┐
                   ├──> data_sync_service ──> PriceData table
Yahoo Finance ─────┘

Sync frequency: Every 30 minutes during market hours
```

### Tick Data Collection
```
Upstox WebSocket ──> tick_collector ──> TickData table

Collection: Continuous (every price tick, multiple per second)
Service: silver-prediction-tick-collector.service
```

### Prediction Generation
```
PriceData ──> add_technical_features() ──> Ensemble.predict() ──> Prediction table
              + sentiment (optional)
              + macro (optional)
```

### Prediction Verification
```
Prediction (pending) ──> get_actual_price() ──> verify() ──> Prediction (verified)
                              │
                              ├── Try tick_data
                              ├── Try price_data
                              └── Try Upstox live quote
```

---

## 5. KEY FILES

| File | Purpose |
|------|---------|
| `backend/workers/scheduler.py` | Primary scheduler (APScheduler) |
| `backend/workers/prediction_verifier.py` | Celery tasks (redundant) |
| `backend/app/services/prediction_engine.py` | Core prediction logic |
| `backend/app/services/data_sync.py` | Data fetching from APIs |
| `backend/app/services/tick_collector.py` | WebSocket tick collection |
| `backend/app/services/upstox_client.py` | Upstox API wrapper |
| `backend/app/models/predictions.py` | Prediction model |
| `backend/app/models/price_data.py` | Price data model |
| `backend/app/models/tick_data.py` | Tick data model |

---

## 6. CONTRACT TYPES

### MCX Silver Contracts
| Contract | Lot Size | Typical Expiry |
|----------|----------|----------------|
| SILVER | 30 kg | Monthly |
| SILVERM | 5 kg | Monthly |
| SILVERMIC | 1 kg | Monthly |

### Contract Selection
- Get all active contracts via `get_all_silver_instrument_keys()`
- Sort by expiry (ascending - nearest first)
- Select one of each type (max 3 contracts)
- Use nearest expiry for each type

---

## 7. MARKET HOURS

### MCX (India)
- **Open**: 9:00 AM IST
- **Close**: 11:30 PM IST
- **Days**: Monday - Friday
- **Holidays**: ~20 Indian market holidays

### COMEX (US)
- **Open**: 6:00 PM Sunday EST
- **Close**: 5:00 PM Friday EST
- **24-hour trading** with 1-hour break (5-6 PM EST daily)

---

## 8. TIMEZONE HANDLING

### Server Time
- Server runs in UTC
- IST = UTC + 5:30

### Database Storage
- All timestamps stored in UTC with timezone info
- Queries use `datetime.utcnow()`

### Display
- Frontend converts to IST for display
- Use `timeZone: 'Asia/Kolkata'` in toLocaleString()

---

## 9. SERVICES (systemd)

| Service | Purpose | Port |
|---------|---------|------|
| silver-prediction-api | FastAPI backend | 8023 |
| silver-prediction-scheduler | APScheduler | - |
| silver-prediction-beat | Celery Beat | - |
| silver-prediction-worker | Celery Worker | - |
| silver-prediction-tick-collector | WebSocket tick collector | - |
| silver-prediction-ws | WebSocket server | - |
| silver-prediction-notifier | Notifications | - |

---

## 10. KNOWN ISSUES

1. **Scheduler Duplication**: Both APScheduler and Celery run same tasks
2. **Verification Uses Wrong Source**: Currently tries price_data first, should use Upstox live quote
3. **Large Tolerance**: 4-hour tolerance for finding actual price is too large for 30m predictions
4. **Token Management**: Upstox token can expire, needs manual refresh

---

## 11. FIXES NEEDED

### Immediate
1. Fix verification to use Upstox live quote for actual price
2. Reduce tolerance in `_get_price_at_time()` to match interval
3. Disable Celery Beat or coordinate with APScheduler

### Future
1. Implement automatic token refresh
2. Add prediction deduplication logic
3. Improve error handling and logging
