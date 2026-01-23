'use client';

import { useEffect, useState } from 'react';
import { getLatestPrediction } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Prediction, Market, Interval, Asset } from '@/lib/types';

interface LatestPredictionsProps {
  asset?: Asset;
  market: Market;
}

const intervals: Interval[] = ['30m', '1h', '4h', '1d'];

const intervalLabels: Record<Interval, string> = {
  '30m': '30m',
  '1h': '1h',
  '4h': '4h',
  '1d': '1D',
};

export default function LatestPredictions({ asset = 'silver', market }: LatestPredictionsProps) {
  const [predictions, setPredictions] = useState<Record<Interval, Prediction | null>>({
    '30m': null,
    '1h': null,
    '4h': null,
    '1d': null,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAllPredictions() {
      setLoading(true);
      const results: Record<Interval, Prediction | null> = {
        '30m': null,
        '1h': null,
        '4h': null,
        '1d': null,
      };

      await Promise.all(
        intervals.map(async (interval) => {
          try {
            const pred = await getLatestPrediction(asset, market, interval);
            results[interval] = pred;
          } catch {
            results[interval] = null;
          }
        })
      );

      setPredictions(results);
      setLoading(false);
    }

    fetchAllPredictions();
    const refreshInterval = setInterval(fetchAllPredictions, 60000); // Refresh every minute
    return () => clearInterval(refreshInterval);
  }, [asset, market]);

  if (loading) {
    return (
      <div className="grid grid-cols-4 gap-2 mt-4">
        {intervals.map((interval) => (
          <div key={interval} className="bg-white/5 rounded-lg p-2 animate-pulse">
            <div className="h-3 w-8 bg-white/10 rounded mb-1"></div>
            <div className="h-4 w-12 bg-white/10 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-4 gap-2 mt-4">
      {intervals.map((interval) => {
        const pred = predictions[interval];
        if (!pred) {
          return (
            <div key={interval} className="bg-white/5 rounded-lg p-2 text-center">
              <div className="text-[10px] text-zinc-500 mb-0.5">{intervalLabels[interval]}</div>
              <div className="text-xs text-zinc-600">No data</div>
            </div>
          );
        }

        const isVerified = pred.verification !== null;
        const isCorrect = pred.verification?.is_direction_correct;

        // Extract contract display (e.g., "SILVERM 27 FEB" from "SILVERM FUT 27 FEB 26")
        const getContractLabel = () => {
          if (!pred.contract_type) return null;
          if (pred.trading_symbol) {
            // Extract expiry from trading symbol (e.g., "SILVERM FUT 27 FEB 26" -> "27 FEB")
            const parts = pred.trading_symbol.split(' ');
            if (parts.length >= 4) {
              return `${pred.contract_type} ${parts[2]} ${parts[3]}`;
            }
          }
          return pred.contract_type;
        };
        const contractLabel = getContractLabel();

        return (
          <div
            key={interval}
            className={cn(
              'rounded-lg p-2 text-center transition-all',
              isVerified
                ? isCorrect
                  ? 'bg-green-500/10 border border-green-500/20'
                  : 'bg-red-500/10 border border-red-500/20'
                : 'bg-white/5 border border-white/5'
            )}
          >
            <div className="text-[10px] text-zinc-500 mb-0.5">
              {intervalLabels[interval]}
              {contractLabel && (
                <span className="block text-[8px] text-zinc-600 truncate" title={pred.trading_symbol || contractLabel}>
                  {contractLabel}
                </span>
              )}
            </div>
            <div className={cn(
              'text-sm font-semibold flex items-center justify-center gap-1',
              pred.predicted_direction === 'bullish' ? 'text-green-400' :
              pred.predicted_direction === 'bearish' ? 'text-red-400' : 'text-zinc-400'
            )}>
              {pred.predicted_direction === 'bullish' ? '↑' : pred.predicted_direction === 'bearish' ? '↓' : '→'}
              <span className="text-xs">
                {(pred.direction_confidence * 100).toFixed(0)}%
              </span>
            </div>
            {isVerified && (
              <div className={cn(
                'text-[9px] mt-0.5',
                isCorrect ? 'text-green-400' : 'text-red-400'
              )}>
                {isCorrect ? '✓' : '✗'}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
