'use client';

import { useEffect, useState } from 'react';
import { getAccuracySummary } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { AccuracySummary, Market, Asset } from '@/lib/types';

const assets: { value: Asset; label: string; icon: string }[] = [
  { value: 'silver', label: 'Silver', icon: '🥈' },
  { value: 'gold', label: 'Gold', icon: '🥇' },
];

const markets: { value: Market; label: string }[] = [
  { value: 'mcx', label: 'MCX (India)' },
  { value: 'comex', label: 'COMEX (US)' },
];

const periods = [
  { value: 7, label: '7 Days' },
  { value: 30, label: '30 Days' },
  { value: 90, label: '90 Days' },
];

export default function AccuracyPage() {
  const [accuracy, setAccuracy] = useState<AccuracySummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedAsset, setSelectedAsset] = useState<Asset>('silver');
  const [selectedMarket, setSelectedMarket] = useState<Market>('mcx');
  const [selectedPeriod, setSelectedPeriod] = useState(30);

  useEffect(() => {
    async function fetchAccuracy() {
      try {
        setLoading(true);
        const data = await getAccuracySummary(selectedAsset, selectedPeriod);
        setAccuracy(data);
      } catch {
        setAccuracy(null);
      } finally {
        setLoading(false);
      }
    }

    fetchAccuracy();
  }, [selectedAsset, selectedMarket, selectedPeriod]);

  const MetricCard = ({
    title,
    value,
    subtitle,
    target,
    status,
  }: {
    title: string;
    value: string | number;
    subtitle?: string;
    target?: string;
    status?: 'good' | 'warning' | 'bad';
  }) => (
    <div className="glass-card p-6">
      <div className="text-xs text-zinc-500 mb-2">{title}</div>
      <div className={cn(
        'text-3xl font-bold mb-1',
        status === 'good' && 'text-green-400',
        status === 'warning' && 'text-yellow-400',
        status === 'bad' && 'text-red-400',
        !status && 'text-white'
      )}>
        {value}
      </div>
      {subtitle && <div className="text-xs text-zinc-500">{subtitle}</div>}
      {target && <div className="text-xs text-zinc-600 mt-1">Target: {target}</div>}
    </div>
  );

  const ProgressBar = ({
    value,
    target,
    label,
  }: {
    value: number;
    target: number;
    label: string;
  }) => {
    const isGood = value >= target;
    return (
      <div className="mb-4">
        <div className="flex items-center justify-between text-sm mb-1">
          <span className="text-zinc-400">{label}</span>
          <span className={cn(
            'font-medium',
            isGood ? 'text-green-400' : 'text-yellow-400'
          )}>
            {value.toFixed(1)}%
          </span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-500',
              isGood ? 'bg-green-500' : 'bg-yellow-500'
            )}
            style={{ width: `${Math.min(value, 100)}%` }}
          />
        </div>
        <div className="text-xs text-zinc-600 mt-1">Target: {target}%</div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="glass-card p-6">
          <div className="skeleton h-8 w-48 rounded mb-2"></div>
          <div className="skeleton h-4 w-96 rounded"></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="glass-card p-6">
              <div className="skeleton h-4 w-24 rounded mb-3"></div>
              <div className="skeleton h-10 w-20 rounded mb-2"></div>
              <div className="skeleton h-3 w-16 rounded"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!accuracy || accuracy.total_predictions === 0) {
    return (
      <div className="space-y-6">
        <div className="glass-card p-6">
          <h1 className="text-2xl font-bold text-white mb-2">Model Accuracy</h1>
          <p className="text-zinc-400">Track prediction performance and confidence interval coverage.</p>
        </div>
        <div className="glass-card p-12 text-center">
          <div className="text-6xl mb-4 opacity-20">📊</div>
          <h2 className="text-xl font-semibold text-white mb-2">No Accuracy Data Yet</h2>
          <p className="text-zinc-400 max-w-md mx-auto">
            Accuracy metrics will be available once predictions are generated and verified against actual prices.
          </p>
        </div>
      </div>
    );
  }

  const directionAccuracy = accuracy.direction_accuracy?.overall ?? 0;
  const ci50 = accuracy.ci_coverage?.ci_50 ?? 0;
  const ci80 = accuracy.ci_coverage?.ci_80 ?? 0;
  const ci95 = accuracy.ci_coverage?.ci_95 ?? 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <h1 className="text-2xl font-bold text-white mb-2">Model Accuracy</h1>
        <p className="text-zinc-400">Track prediction performance and confidence interval coverage.</p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        {/* Asset Filter */}
        <div className="glass-card p-1 flex">
          {assets.map((asset) => (
            <button
              key={asset.value}
              onClick={() => setSelectedAsset(asset.value)}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5',
                selectedAsset === asset.value
                  ? 'bg-gradient-to-r from-amber-500/20 to-yellow-500/20 text-amber-400'
                  : 'text-zinc-400 hover:text-white'
              )}
            >
              <span>{asset.icon}</span>
              <span>{asset.label}</span>
            </button>
          ))}
        </div>

        <div className="glass-card p-1 flex">
          {markets.map((market) => (
            <button
              key={market.value}
              onClick={() => setSelectedMarket(market.value)}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-all',
                selectedMarket === market.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-zinc-400 hover:text-white'
              )}
            >
              {market.label}
            </button>
          ))}
        </div>

        <div className="glass-card p-1 flex">
          {periods.map((period) => (
            <button
              key={period.value}
              onClick={() => setSelectedPeriod(period.value)}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-all',
                selectedPeriod === period.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-zinc-400 hover:text-white'
              )}
            >
              {period.label}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Direction Accuracy"
          value={`${directionAccuracy.toFixed(1)}%`}
          subtitle={`${accuracy.direction_accuracy?.correct ?? 0} correct predictions`}
          target=">55%"
          status={directionAccuracy >= 55 ? 'good' : directionAccuracy >= 50 ? 'warning' : 'bad'}
        />
        <MetricCard
          title="Total Predictions"
          value={accuracy.total_predictions}
          subtitle={`${accuracy.verified_predictions} verified`}
        />
        <MetricCard
          title="Mean Absolute Error"
          value={`${(accuracy.error_metrics?.mae ?? 0).toFixed(2)}`}
          subtitle="Price units"
        />
        <MetricCard
          title="MAPE"
          value={`${(accuracy.error_metrics?.mape ?? 0).toFixed(2)}%`}
          subtitle="Mean Absolute Percentage Error"
          status={(accuracy.error_metrics?.mape ?? 100) < 2 ? 'good' : (accuracy.error_metrics?.mape ?? 100) < 5 ? 'warning' : 'bad'}
        />
      </div>

      {/* Confidence Interval Coverage */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-6">Confidence Interval Coverage</h2>
        <div className="max-w-2xl">
          <ProgressBar value={ci50} target={50} label="50% CI Coverage" />
          <ProgressBar value={ci80} target={80} label="80% CI Coverage" />
          <ProgressBar value={ci95} target={95} label="95% CI Coverage" />
        </div>
        <p className="text-xs text-zinc-600 mt-4">
          A well-calibrated model should have actual prices falling within confidence intervals at rates matching the CI level.
        </p>
      </div>

      {/* Streaks */}
      {accuracy.streaks && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Prediction Streaks</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-white/5 rounded-xl">
              <div className="text-xs text-zinc-500 mb-2">Current Streak</div>
              <div className={cn(
                'text-2xl font-bold',
                accuracy.streaks.current?.type === 'win' ? 'text-green-400' : 'text-red-400'
              )}>
                {accuracy.streaks.current?.count ?? 0} {accuracy.streaks.current?.type === 'win' ? 'Wins' : 'Losses'}
              </div>
            </div>
            <div className="text-center p-4 bg-white/5 rounded-xl">
              <div className="text-xs text-zinc-500 mb-2">Best Win Streak</div>
              <div className="text-2xl font-bold text-green-400">
                {accuracy.streaks.best_win ?? 0}
              </div>
            </div>
            <div className="text-center p-4 bg-white/5 rounded-xl">
              <div className="text-xs text-zinc-500 mb-2">Worst Loss Streak</div>
              <div className="text-2xl font-bold text-red-400">
                {accuracy.streaks.worst_loss ?? 0}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* By Interval */}
      {accuracy.by_interval && Object.keys(accuracy.by_interval).length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Performance by Interval</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs text-zinc-500 font-medium p-3">Interval</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-3">Predictions</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-3">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(accuracy.by_interval).map(([interval, data]) => (
                  <tr key={interval} className="border-b border-white/5">
                    <td className="p-3 text-sm text-white uppercase">{interval}</td>
                    <td className="p-3 text-sm text-zinc-300 text-right">{data.predictions}</td>
                    <td className="p-3 text-right">
                      <span className={cn(
                        'text-sm font-medium',
                        data.accuracy >= 55 ? 'text-green-400' : 'text-yellow-400'
                      )}>
                        {data.accuracy.toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="glass-card p-4 border-yellow-500/20">
        <div className="flex items-start gap-3">
          <span className="text-yellow-400 text-xl">⚠️</span>
          <div>
            <div className="text-sm font-medium text-yellow-400 mb-1">Disclaimer</div>
            <p className="text-xs text-zinc-400">
              Past performance does not guarantee future results. Predictions are probabilistic and should be used as one input among many in your decision-making process. This is a decision support tool, not financial advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
