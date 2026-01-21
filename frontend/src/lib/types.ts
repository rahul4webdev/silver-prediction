/**
 * Type definitions for the Silver Prediction System
 */

export interface Prediction {
  id: string;
  created_at: string;
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
  confidence_intervals?: {
    ci_50: { lower: number; upper: number };
    ci_80: { lower: number; upper: number };
    ci_95: { lower: number; upper: number };
  };
  model_weights: Record<string, number>;
  model_version?: string;
  actual_price?: number | null;
  is_direction_correct?: boolean | null;
  price_error?: number | null;
  price_error_percent?: number | null;
  within_ci_50?: boolean | null;
  within_ci_80?: boolean | null;
  within_ci_95?: boolean | null;
  verified_at?: string | null;
  verification?: {
    actual_price: number | null;
    is_direction_correct: boolean | null;
    price_error: number | null;
    price_error_percent: number | null;
    within_ci_50: boolean | null;
    within_ci_80: boolean | null;
    within_ci_95: boolean | null;
    verified_at: string | null;
  } | null;
}

export interface PriceCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface AccuracySummary {
  total_predictions: number;
  verified_predictions: number;
  pending_verification?: number;
  period_days?: number;
  direction_accuracy: {
    overall: number;
    correct: number;
    wrong?: number;
    bullish_accuracy?: number;
    bearish_accuracy?: number;
    trend?: number[];
  };
  ci_coverage: {
    ci_50: number;
    ci_80: number;
    ci_95: number;
  };
  confidence_interval_coverage?: {
    ci_50: number;
    ci_80: number;
    ci_95: number;
  };
  error_metrics: {
    mae: number;
    mape: number;
    rmse: number;
  };
  by_interval: Record<string, { predictions: number; accuracy: number; total?: number; correct?: number }>;
  by_market: Record<string, { predictions: number; accuracy: number; total?: number; correct?: number }>;
  streaks: {
    current: { type: 'win' | 'loss'; count: number };
    best_win: number;
    worst_loss: number;
  };
}

export interface HealthStatus {
  status: string;
  timestamp?: string;
  environment?: string;
  database?: string;
  has_token?: boolean;
  trained_models?: number;
  models?: Record<string, {
    is_trained: boolean;
    last_trained: string | null;
    weights: Record<string, number>;
  }>;
}

export interface ModelInfo {
  interval: string;
  is_trained: boolean;
  last_trained: string | null;
  weights: Record<string, number>;
  models: Record<string, {
    name: string;
    is_trained: boolean;
    last_trained: string | null;
    training_metrics: Record<string, number>;
  }>;
}

export type Market = 'mcx' | 'comex';
export type Interval = '30m' | '1h' | '4h' | 'daily';
export type Direction = 'bullish' | 'bearish' | 'neutral';
