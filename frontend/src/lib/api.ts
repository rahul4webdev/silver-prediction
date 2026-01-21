/**
 * API client for the Silver Prediction System
 */

import axios from 'axios';
import type {
  Prediction,
  PriceCandle,
  AccuracySummary,
  HealthStatus,
  ModelInfo,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health
export async function getHealth(): Promise<HealthStatus> {
  const response = await api.get('/health');
  return response.data;
}

export async function getModelsHealth(): Promise<HealthStatus> {
  const response = await api.get('/health/models');
  return response.data;
}

// Predictions
export async function generatePrediction(
  asset: string = 'silver',
  market: string = 'mcx',
  interval: string = '30m'
): Promise<{ status: string; prediction: Prediction }> {
  const response = await api.post('/predictions/generate', null, {
    params: { asset, market, interval },
  });
  return response.data;
}

export async function getLatestPrediction(
  asset: string = 'silver',
  market?: string,
  interval?: string
): Promise<Prediction> {
  const response = await api.get('/predictions/latest', {
    params: { asset, market, interval },
  });
  return response.data;
}

export async function getPredictionHistory(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  verifiedOnly: boolean = false,
  limit: number = 100
): Promise<{ total: number; predictions: Prediction[] }> {
  const response = await api.get('/predictions/history', {
    params: {
      asset,
      market,
      interval,
      verified_only: verifiedOnly,
      limit,
    },
  });
  return response.data;
}

export async function triggerVerification(): Promise<{ status: string; result: any }> {
  const response = await api.post('/predictions/verify');
  return response.data;
}

export async function trainModels(
  asset: string = 'silver',
  market: string = 'mcx',
  interval: string = '30m'
): Promise<{ status: string; result: any }> {
  const response = await api.post('/predictions/train', null, {
    params: { asset, market, interval },
  });
  return response.data;
}

export async function getModelInfo(interval: string = '30m'): Promise<ModelInfo> {
  const response = await api.get('/predictions/model/info', {
    params: { interval },
  });
  return response.data;
}

// Historical Data
export async function getHistoricalData(
  asset: string,
  market: string,
  interval: string = '30m',
  limit: number = 100
): Promise<{ candles: PriceCandle[] }> {
  const response = await api.get(`/historical/${asset}/${market}`, {
    params: { interval, limit },
  });
  return response.data;
}

export async function getLatestPrice(
  asset: string,
  market: string,
  interval: string = '30m'
): Promise<PriceCandle> {
  const response = await api.get(`/historical/${asset}/${market}/latest`, {
    params: { interval },
  });
  return response.data;
}

export async function getPriceStats(
  asset: string,
  market: string,
  interval: string = '30m',
  periodDays: number = 30
): Promise<any> {
  const response = await api.get(`/historical/${asset}/${market}/stats`, {
    params: { interval, period_days: periodDays },
  });
  return response.data;
}

// Accuracy
export async function getAccuracySummary(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  periodDays: number = 30
): Promise<AccuracySummary> {
  const response = await api.get('/accuracy/summary', {
    params: { asset, market, interval, period_days: periodDays },
  });
  return response.data;
}

export async function getAccuracyTrend(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  periodDays: number = 30,
  bucketDays: number = 1
): Promise<any> {
  const response = await api.get('/accuracy/trend', {
    params: {
      asset,
      market,
      interval,
      period_days: periodDays,
      bucket_days: bucketDays,
    },
  });
  return response.data;
}

export async function getCICoverage(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  periodDays: number = 30
): Promise<any> {
  const response = await api.get('/accuracy/confidence-intervals', {
    params: { asset, market, interval, period_days: periodDays },
  });
  return response.data;
}

export async function getStreaks(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  periodDays: number = 30
): Promise<any> {
  const response = await api.get('/accuracy/streaks', {
    params: { asset, market, interval, period_days: periodDays },
  });
  return response.data;
}

// Convenience wrappers for frontend pages
export async function getPriceData(
  asset: string,
  market: string,
  interval: string,
  limit: number = 500
): Promise<PriceCandle[]> {
  const response = await getHistoricalData(asset, market, interval, limit);
  return response.candles || [];
}

export async function getPredictions(
  asset: string = 'silver',
  market?: string,
  interval?: string,
  limit: number = 100,
  verifiedOnly: boolean = false
): Promise<Prediction[]> {
  const response = await getPredictionHistory(asset, market, interval, verifiedOnly, limit);
  return response.predictions || [];
}

export async function triggerPrediction(
  asset: string,
  market: string,
  interval: string
): Promise<Prediction | null> {
  const response = await generatePrediction(asset, market, interval);
  return response.prediction || null;
}

// Export default for easy importing
export default api;
