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

export interface ConfidenceInterval {
  lower: number;
  upper: number;
}

export interface ConfidenceIntervals {
  ci_50: ConfidenceInterval;
  ci_80: ConfidenceInterval;
  ci_95: ConfidenceInterval;
}

export interface PredictionVerification {
  actual_price: number;
  is_direction_correct: boolean;
  price_error: number;
  price_error_percent: number;
  within_ci_50: boolean;
  within_ci_80: boolean;
  within_ci_95: boolean;
  verified_at: string;
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
  confidence_intervals: ConfidenceIntervals;
  model_weights?: Record<string, number>;
  model_version?: string | null;
  verification: PredictionVerification | null;
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

export type Asset = 'silver' | 'gold';
export type Market = 'mcx' | 'comex';
export type Interval = '30m' | '1h' | '4h' | '1d';
export type ContractType = 'SILVER' | 'SILVERM' | 'SILVERMIC';

export interface ContractInfo {
  instrument_key: string;
  trading_symbol: string;
  contract_type: ContractType;
  expiry: string | null;
  expiry_date: string | null;  // Human readable format like "27 Feb 2026"
  lot_size: number | null;
  tick_size: number | null;
  is_active: boolean;
}

export interface ContractsResponse {
  status: string;
  contracts: ContractInfo[];
  total: number;
  default_contract: string | null;
}
