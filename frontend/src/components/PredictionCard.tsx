'use client';

import { useEffect, useState } from 'react';
import { getLatestPrediction } from '@/lib/api';
import { formatCurrency, formatPercent, formatDateTime, cn } from '@/lib/utils';
import type { Prediction, Market, Interval } from '@/lib/types';

interface PredictionCardProps {
  market: Market;
  interval?: Interval;
}

export default function PredictionCard({ market, interval }: PredictionCardProps) {
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPrediction() {
      try {
        setLoading(true);
        const data = await getLatestPrediction('silver', market, interval);
        setPrediction(data);
      } catch {
        setPrediction(null);
      } finally {
        setLoading(false);
      }
    }

    fetchPrediction();
  }, [market, interval]);

  const currency = market === 'mcx' ? 'INR' : 'USD';

  if (loading) {
    return (
      <div className="glass-card p-6">
        <div className="skeleton h-4 w-32 rounded mb-4"></div>
        <div className="skeleton h-8 w-24 rounded mb-4"></div>
        <div className="skeleton h-4 w-full rounded mb-2"></div>
        <div className="skeleton h-4 w-3/4 rounded"></div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="glass-card p-6">
        <div className="text-zinc-400 text-sm mb-2">Latest Prediction</div>
        <div className="text-zinc-500">No predictions available yet</div>
        <p className="text-xs text-zinc-600 mt-2">
          Predictions will appear once the ML models are trained with historical data.
        </p>
      </div>
    );
  }

  const isBullish = prediction.predicted_direction === 'bullish';
  const confidence = prediction.direction_confidence * 100;

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <span className="text-zinc-400 text-sm font-medium">Latest Prediction</span>
        <span className="text-xs text-zinc-500">{prediction.interval}</span>
      </div>

      {/* Direction Badge */}
      <div className={cn(
        "inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold mb-4",
        isBullish
          ? "bg-green-500/20 text-green-400 border border-green-500/30"
          : "bg-red-500/20 text-red-400 border border-red-500/30"
      )}>
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          {isBullish ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          )}
        </svg>
        {prediction.predicted_direction.toUpperCase()}
      </div>

      {/* Confidence */}
      <div className="mb-4">
        <div className="flex items-center justify-between text-sm mb-1">
          <span className="text-zinc-400">Confidence</span>
          <span className="text-cyan-400 font-semibold">{confidence.toFixed(1)}%</span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all"
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      {/* Price targets */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <div className="text-zinc-500 text-xs">Current</div>
          <div className="text-white font-medium">
            {formatCurrency(prediction.current_price, currency)}
          </div>
        </div>
        <div>
          <div className="text-zinc-500 text-xs">Predicted</div>
          <div className={cn(
            "font-medium",
            isBullish ? "text-green-400" : "text-red-400"
          )}>
            {formatCurrency(prediction.predicted_price, currency)}
          </div>
        </div>
      </div>

      {/* Target time */}
      <div className="mt-4 pt-4 border-t border-white/5">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-500">Target Time</span>
          <span className="text-zinc-300">{formatDateTime(prediction.target_time)}</span>
        </div>
      </div>
    </div>
  );
}
