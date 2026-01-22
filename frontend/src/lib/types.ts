/**
 * Type definitions for Silver Prediction System
 */

export interface PriceCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface LivePrice {
  asset: string;
  market: string;
  symbol?: string;
  price: number;
  open?: number;
  high?: number;
  low?: number;
  previous_close?: number;
  change?: number;
  change_percent?: number;
  volume?: number;
  timestamp: string;
  source: string;
  currency?: string;
  status?: string;
  message?: string;
}

export interface HistoricalData {
  asset: string;
  market: string;
  interval: string;
  count: number;
  candles: PriceCandle[];
  source: string;
  message?: string;
}

export interface Prediction {
  id?: string;
  created_at?: string;
  asset: string;
  market: string;
  interval: string;
  prediction_time: string;
  target_time: string;
  current_price: number;
  predicted_price: number;
  predicted_direction: 'bullish' | 'bearish' | 'neutral';
  direction_confidence: number;
  ci_50_lower?: number;
  ci_50_upper?: number;
  ci_80_lower?: number;
  ci_80_upper?: number;
  ci_95_lower?: number;
  ci_95_upper?: number;
  model_weights?: Record<string, number>;
  actual_price?: number | null;
  is_direction_correct?: boolean | null;
  verified_at?: string | null;
}

export interface AccuracySummary {
  total_predictions: number;
  verified_predictions: number;
  pending_verification?: number;
  direction_accuracy: {
    overall: number;
    correct?: number;
  };
  ci_coverage: {
    ci_50: number;
    ci_80: number;
    ci_95: number;
  };
  error_metrics: {
    mae: number;
    mape: number;
    rmse: number;
  };
  by_interval?: Record<string, { predictions: number; accuracy: number }>;
  by_market?: Record<string, { predictions: number; accuracy: number }>;
  streaks?: {
    current: { type: 'win' | 'loss'; count: number };
    best_win: number;
    worst_loss: number;
  };
}

export type Market = 'mcx' | 'comex';
export type Interval = '30m' | '1h' | '4h' | '1d';
