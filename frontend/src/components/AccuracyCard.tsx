'use client';

import { useEffect, useState } from 'react';
import { getAccuracySummary } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { AccuracySummary } from '@/lib/types';

export default function AccuracyCard() {
  const [accuracy, setAccuracy] = useState<AccuracySummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAccuracy() {
      try {
        setLoading(true);
        const data = await getAccuracySummary('silver', 30);
        setAccuracy(data);
      } catch {
        setAccuracy(null);
      } finally {
        setLoading(false);
      }
    }

    fetchAccuracy();
  }, []);

  if (loading) {
    return (
      <div className="glass-card p-6">
        <div className="skeleton h-4 w-32 rounded mb-4"></div>
        <div className="grid grid-cols-3 gap-4">
          <div className="skeleton h-16 rounded"></div>
          <div className="skeleton h-16 rounded"></div>
          <div className="skeleton h-16 rounded"></div>
        </div>
      </div>
    );
  }

  if (!accuracy || accuracy.total_predictions === 0) {
    return (
      <div className="glass-card p-6">
        <div className="text-zinc-400 text-sm mb-2">Model Accuracy</div>
        <div className="text-zinc-500">No accuracy data yet</div>
        <p className="text-xs text-zinc-600 mt-2">
          Accuracy metrics will be available after predictions are verified.
        </p>
      </div>
    );
  }

  const directionAccuracy = accuracy.direction_accuracy?.overall ?? 0;
  const ci80Coverage = accuracy.ci_coverage?.ci_80 ?? 0;

  return (
    <div className="glass-card p-6">
      <div className="flex items-center justify-between mb-4">
        <span className="text-zinc-400 text-sm font-medium">Model Accuracy</span>
        <span className="text-xs text-zinc-500">Last 30 days</span>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Direction Accuracy */}
        <div className="text-center">
          <div className={cn(
            "text-2xl font-bold mb-1",
            directionAccuracy >= 55 ? "text-green-400" : "text-yellow-400"
          )}>
            {directionAccuracy.toFixed(1)}%
          </div>
          <div className="text-xs text-zinc-500">Direction</div>
        </div>

        {/* 80% CI Coverage */}
        <div className="text-center">
          <div className={cn(
            "text-2xl font-bold mb-1",
            ci80Coverage >= 75 ? "text-green-400" : "text-yellow-400"
          )}>
            {ci80Coverage.toFixed(1)}%
          </div>
          <div className="text-xs text-zinc-500">80% CI</div>
        </div>

        {/* Total Predictions */}
        <div className="text-center">
          <div className="text-2xl font-bold text-cyan-400 mb-1">
            {accuracy.verified_predictions}
          </div>
          <div className="text-xs text-zinc-500">Verified</div>
        </div>
      </div>

      {/* Streak info if available */}
      {accuracy.streaks?.current && (
        <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between">
          <span className="text-xs text-zinc-500">Current Streak</span>
          <span className={cn(
            "text-sm font-medium",
            accuracy.streaks.current.type === 'win' ? "text-green-400" : "text-red-400"
          )}>
            {accuracy.streaks.current.count} {accuracy.streaks.current.type === 'win' ? 'Wins' : 'Losses'}
          </span>
        </div>
      )}
    </div>
  );
}
