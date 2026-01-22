/**
 * API client for Silver Prediction System
 */

import type {
  Prediction,
  PriceCandle,
  AccuracySummary,
  LivePrice,
  HistoricalData,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://predictionapi.gahfaudio.in';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_URL}/api/v1${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function getLivePrice(asset: string, market: string): Promise<LivePrice> {
  const params = new URLSearchParams({ market });
  return fetchAPI<LivePrice>(`/historical/live/${asset}?${params}`);
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
