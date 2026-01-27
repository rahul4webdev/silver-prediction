/**
 * API client for Silver Prediction System
 */

import type {
  Prediction,
  PriceCandle,
  AccuracySummary,
  LivePrice,
  HistoricalData,
  ContractsResponse,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://predictionapi.gahfaudio.in';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_URL}/api/v1${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'No error details');
      console.error(`API error for ${endpoint}: ${response.status} ${response.statusText}`, errorText);
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      console.error(`Network error for ${endpoint}: CORS or connectivity issue`, error);
      throw new Error(`Network error: Unable to reach API at ${url}`);
    }
    throw error;
  }
}

export async function getLivePrice(asset: string, market: string): Promise<LivePrice> {
  const params = new URLSearchParams({ market });
  // Add cache: no-store to ensure fresh data on every request
  return fetchAPI<LivePrice>(`/historical/live/${asset}?${params}`, { cache: 'no-store' });
}

export async function getHistoricalData(
  asset: string,
  market: string,
  interval: string = '1d',
  limit: number = 100
): Promise<HistoricalData> {
  const params = new URLSearchParams({ interval, limit: limit.toString() });
  return fetchAPI<HistoricalData>(`/historical/${asset}/${market}?${params}`);
}

export async function getLatestPrediction(
  asset: string = 'silver',
  market?: string,
  interval?: string
): Promise<Prediction | null> {
  const params = new URLSearchParams({ asset });
  if (market) params.set('market', market);
  if (interval) params.set('interval', interval);

  try {
    const data = await fetchAPI<Prediction & { status?: string }>(`/predictions/latest?${params}`);
    if (data.status === 'no_predictions' || data.status === 'error') {
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

export async function getPredictionHistory(
  asset: string = 'silver',
  limit: number = 50
): Promise<{ total: number; predictions: Prediction[] }> {
  const params = new URLSearchParams({ asset, limit: limit.toString() });
  try {
    return await fetchAPI<{ total: number; predictions: Prediction[] }>(`/predictions/history?${params}`);
  } catch {
    return { total: 0, predictions: [] };
  }
}

export async function getPredictions(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  limit: number = 100
): Promise<Prediction[]> {
  const params = new URLSearchParams({ asset, limit: limit.toString() });
  if (market) params.set('market', market);
  if (interval && interval !== 'all') params.set('interval', interval);
  try {
    const data = await fetchAPI<{ total: number; predictions: Prediction[] }>(`/predictions/history?${params}`);
    return data.predictions || [];
  } catch {
    return [];
  }
}

export async function getAccuracySummary(
  asset: string = 'silver',
  periodDays: number = 30
): Promise<AccuracySummary> {
  const params = new URLSearchParams({ asset, period_days: periodDays.toString() });
  try {
    return await fetchAPI<AccuracySummary>(`/accuracy/summary?${params}`);
  } catch {
    return {
      total_predictions: 0,
      verified_predictions: 0,
      direction_accuracy: { overall: 0 },
      ci_coverage: { ci_50: 0, ci_80: 0, ci_95: 0 },
      error_metrics: { mae: 0, mape: 0, rmse: 0 },
    };
  }
}

export async function getPriceData(
  asset: string,
  market: string,
  interval: string,
  limit: number = 100
): Promise<PriceCandle[]> {
  try {
    const data = await getHistoricalData(asset, market, interval, limit);
    return data.candles || [];
  } catch {
    return [];
  }
}

export interface SentimentData {
  status: string;
  asset: string;
  sentiment: {
    overall: number;
    label: string;
    confidence: number;
  };
  stats: {
    article_count: number;
    bullish_count: number;
    bearish_count: number;
    neutral_count: number;
    average_relevance: number;
  };
  timestamp: string;
  articles: Array<{
    title: string;
    source: string;
    published_at: string;
    url: string;
    sentiment_score: number | null;
    relevance_score: number | null;
  }>;
}

export async function getSentiment(
  asset: string = 'silver',
  lookbackDays: number = 3
): Promise<SentimentData | null> {
  const params = new URLSearchParams({ asset, lookback_days: lookbackDays.toString() });
  try {
    const data = await fetchAPI<SentimentData>(`/sentiment/current?${params}`);
    if (data.status === 'error') {
      return null;
    }
    return data;
  } catch {
    return null;
  }
}

// ============================================================
// Macro Data
// ============================================================

export interface MacroData {
  status: string;
  timestamp: string;
  dxy: {
    value: number;
    change_1d: number;
    change_1w: number;
    trend: string;
  };
  fear_greed: {
    value: number;
    label: string;
    previous_value: number;
    change: number;
  };
  cot: {
    net_positions: number;
    commercial_long: number;
    commercial_short: number;
    non_commercial_long: number;
    non_commercial_short: number;
    change_week: number;
  };
  treasury: {
    us_10y: number;
    us_2y: number;
    spread_10y_2y: number;
  };
}

export async function getMacroData(asset: string = 'silver'): Promise<MacroData | null> {
  try {
    return await fetchAPI<MacroData>(`/macro/all?asset=${asset}`);
  } catch {
    return null;
  }
}

// ============================================================
// Alerts
// ============================================================

export interface PriceAlert {
  id: string;
  asset: string;
  market: string;
  alert_type: 'above' | 'below';
  target_price: number;
  current_price_at_creation: number;
  status: 'active' | 'triggered' | 'cancelled';
  created_at: string;
  triggered_at: string | null;
  notes: string | null;
}

export async function createAlert(
  asset: string,
  market: string,
  alertType: 'above' | 'below',
  targetPrice: number,
  notes?: string
): Promise<PriceAlert | null> {
  try {
    return await fetchAPI<PriceAlert>('/alerts/create', {
      method: 'POST',
      body: JSON.stringify({
        asset,
        market,
        alert_type: alertType,
        target_price: targetPrice,
        notes,
      }),
    });
  } catch {
    return null;
  }
}

export async function getAlerts(
  asset?: string,
  status?: string
): Promise<{ alerts: PriceAlert[]; total: number }> {
  const params = new URLSearchParams();
  if (asset) params.set('asset', asset);
  if (status) params.set('status', status);
  try {
    return await fetchAPI<{ alerts: PriceAlert[]; total: number }>(`/alerts/list?${params}`);
  } catch {
    return { alerts: [], total: 0 };
  }
}

export async function deleteAlert(alertId: string): Promise<boolean> {
  try {
    await fetchAPI(`/alerts/${alertId}`, { method: 'DELETE' });
    return true;
  } catch {
    return false;
  }
}

// ============================================================
// Trade Journal
// ============================================================

export interface Trade {
  id: string;
  asset: string;
  market: string;
  trade_type: 'long' | 'short';
  entry_price: number;
  exit_price: number | null;
  quantity: number;
  entry_time: string;
  exit_time: string | null;
  pnl: number | null;
  pnl_percent: number | null;
  status: 'open' | 'closed';
  prediction_id: string | null;
  followed_prediction: boolean;
  notes: string | null;
}

export async function createTrade(data: {
  asset: string;
  market: string;
  trade_type: 'long' | 'short';
  entry_price: number;
  quantity: number;
  prediction_id?: string;
  followed_prediction?: boolean;
  notes?: string;
}): Promise<Trade | null> {
  try {
    return await fetchAPI<Trade>('/alerts/trades/create', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  } catch {
    return null;
  }
}

export async function closeTrade(
  tradeId: string,
  exitPrice: number,
  notes?: string
): Promise<Trade | null> {
  try {
    return await fetchAPI<Trade>(`/alerts/trades/${tradeId}/close`, {
      method: 'POST',
      body: JSON.stringify({ exit_price: exitPrice, notes }),
    });
  } catch {
    return null;
  }
}

export async function getTrades(
  asset?: string,
  status?: string,
  limit: number = 50
): Promise<{ trades: Trade[]; total: number }> {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (asset) params.set('asset', asset);
  if (status) params.set('status', status);
  try {
    return await fetchAPI<{ trades: Trade[]; total: number }>(`/alerts/trades/list?${params}`);
  } catch {
    return { trades: [], total: 0 };
  }
}

export interface TradePerformance {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  average_pnl: number;
  best_trade: number;
  worst_trade: number;
  followed_prediction_accuracy: number;
}

export async function getTradePerformance(
  asset?: string,
  days: number = 30
): Promise<TradePerformance | null> {
  const params = new URLSearchParams({ days: days.toString() });
  if (asset) params.set('asset', asset);
  try {
    return await fetchAPI<TradePerformance>(`/alerts/trades/performance?${params}`);
  } catch {
    return null;
  }
}

// ============================================================
// Confluence
// ============================================================

export interface ConfluenceData {
  status: string;
  asset: string;
  market: string;
  timestamp: string;
  confluence_detected: boolean;
  direction: 'bullish' | 'bearish' | 'mixed';
  alignment_ratio: number;
  aligned_intervals: string[];
  total_intervals: number;
  avg_confidence: number;
  strength: number;
  signal_quality: 'strong' | 'moderate' | 'weak' | 'none';
  predictions: Record<string, {
    direction: string;
    confidence: number;
    predicted_price: number;
    current_price: number;
    target_time: string;
    interval: string;
  } | null>;
  recommendation: string;
}

export async function getConfluence(
  asset: string = 'silver',
  market: string = 'mcx'
): Promise<ConfluenceData | null> {
  const params = new URLSearchParams({ asset, market });
  try {
    return await fetchAPI<ConfluenceData>(`/confluence/detect?${params}`);
  } catch {
    return null;
  }
}

// ============================================================
// Correlations
// ============================================================

export interface CorrelationData {
  status: string;
  lookback_days: number;
  timestamp: string;
  correlations: Record<string, {
    correlation: number;
    expected: string;
    typical: number;
    status: string;
    data_points: number;
  }>;
  market_regime: {
    type: string;
    description: string;
  };
}

export async function getCorrelations(
  lookbackDays: number = 30
): Promise<CorrelationData | null> {
  const params = new URLSearchParams({ lookback_days: lookbackDays.toString() });
  try {
    return await fetchAPI<CorrelationData>(`/confluence/correlations?${params}`);
  } catch {
    return null;
  }
}

// ============================================================
// Contracts
// ============================================================

export async function getMCXSilverContracts(
  includeExpired: boolean = false
): Promise<ContractsResponse | null> {
  const params = new URLSearchParams({ include_expired: includeExpired.toString() });
  try {
    return await fetchAPI<ContractsResponse>(`/contracts/mcx/silver?${params}`);
  } catch {
    return null;
  }
}

// ============================================================
// System Status
// ============================================================

export interface ServiceStatus {
  status: string;
  connected?: boolean;
  authenticated?: boolean;
  is_running?: boolean;
  error?: string;
  [key: string]: unknown;
}

export interface ModelStatus {
  is_trained: boolean;
  models?: Record<string, { is_trained: boolean; weight: number }>;
  error?: string;
}

export interface SchedulerJob {
  id: string;
  name: string;
  next_run?: string | null;
  schedule?: string;
}

export interface SystemStatus {
  timestamp: string;
  environment: string;
  overall_health: string;
  services: Record<string, ServiceStatus>;
  models: Record<string, ModelStatus>;
  scheduler: {
    status: string;
    is_running: boolean;
    jobs: SchedulerJob[];
  };
  database: {
    predictions: {
      total: number;
      verified: number;
      pending: number;
      today: number;
    };
  };
  market: {
    current_time_ist: string;
    is_weekend: boolean;
    is_trading_hours: boolean;
    market_status: string;
    hours: {
      open: string;
      close: string;
    };
  };
}

export async function getSystemStatus(): Promise<SystemStatus | null> {
  try {
    return await fetchAPI<SystemStatus>('/status/');
  } catch {
    return null;
  }
}

export async function getModelsStatus(): Promise<{ status: string; intervals: Record<string, ModelStatus> } | null> {
  try {
    return await fetchAPI<{ status: string; intervals: Record<string, ModelStatus> }>('/status/models');
  } catch {
    return null;
  }
}

export async function getPredictionsSummary(): Promise<{
  total_predictions: number;
  verified_predictions: number;
  pending_predictions: number;
  correct_predictions: number;
  accuracy_percent: number;
  predictions_today: number;
} | null> {
  try {
    return await fetchAPI('/status/predictions/summary');
  } catch {
    return null;
  }
}
