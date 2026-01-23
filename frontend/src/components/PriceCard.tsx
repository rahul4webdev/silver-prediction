'use client';

import { useEffect, useState } from 'react';
import { getLivePrice } from '@/lib/api';
import { formatCurrency, formatPercent, cn } from '@/lib/utils';
import type { LivePrice, Asset } from '@/lib/types';
import LatestPredictions from './LatestPredictions';

interface PriceCardProps {
  asset?: Asset;
  market: 'mcx' | 'comex';
}

export default function PriceCard({ asset = 'silver', market }: PriceCardProps) {
  const [price, setPrice] = useState<LivePrice | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPrice() {
      try {
        setLoading(true);
        const data = await getLivePrice(asset, market);
        if (data.status === 'error') {
          setError(data.message || 'Failed to fetch price');
        } else {
          setPrice(data);
          setError(null);
        }
      } catch (err) {
        setError('Failed to fetch price');
      } finally {
        setLoading(false);
      }
    }

    fetchPrice();
    const interval = setInterval(fetchPrice, 5000); // Refresh every 5 seconds for more real-time feel
    return () => clearInterval(interval);
  }, [asset, market]);

  const isPositive = (price?.change_percent ?? 0) >= 0;
  const currency = market === 'mcx' ? 'INR' : 'USD';

  if (loading) {
    return (
      <div className="glass-card p-4 sm:p-6">
        <div className="skeleton h-4 w-24 rounded mb-4"></div>
        <div className="skeleton h-8 sm:h-10 w-32 sm:w-40 rounded mb-2"></div>
        <div className="skeleton h-4 w-20 rounded"></div>
        <div className="grid grid-cols-4 gap-2 mt-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-white/5 rounded-lg p-2 animate-pulse">
              <div className="h-3 w-6 sm:w-8 bg-white/10 rounded mb-1 mx-auto"></div>
              <div className="h-4 w-10 sm:w-12 bg-white/10 rounded mx-auto"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !price) {
    return (
      <div className="glass-card p-4 sm:p-6">
        <div className="text-zinc-400 text-sm">{market.toUpperCase()} {asset.charAt(0).toUpperCase() + asset.slice(1)}</div>
        <div className="text-zinc-500 mt-2">Unable to load price</div>
        <LatestPredictions asset={asset} market={market} />
      </div>
    );
  }

  return (
    <div className={cn(
      "glass-card p-4 sm:p-6 relative overflow-hidden",
      isPositive ? "glow-success" : "glow-danger"
    )}>
      {/* Background gradient */}
      <div className={cn(
        "absolute inset-0 opacity-10",
        isPositive
          ? "bg-gradient-to-br from-green-500 to-transparent"
          : "bg-gradient-to-br from-red-500 to-transparent"
      )} />

      <div className="relative">
        <div className="flex items-center justify-between mb-2 sm:mb-3">
          <span className="text-zinc-400 text-xs sm:text-sm font-medium">{market.toUpperCase()} {asset.charAt(0).toUpperCase() + asset.slice(1)}</span>
          <span className="text-[10px] sm:text-xs text-zinc-500 bg-white/5 px-2 py-1 rounded truncate max-w-[80px] sm:max-w-none">
            {price.source?.replace('_', ' ')}
          </span>
        </div>

        <div className="flex items-baseline gap-2 sm:gap-3">
          <div className="text-2xl sm:text-3xl font-bold text-white">
            {formatCurrency(price.price, currency)}
          </div>
          <div className={cn(
            "text-xs sm:text-sm font-medium",
            isPositive ? "text-bullish" : "text-bearish"
          )}>
            {price.change_percent !== undefined && formatPercent(price.change_percent)}
          </div>
        </div>

        {market === 'mcx' && (
          <div className="text-[10px] sm:text-xs text-zinc-500 mt-1">
            Price per kg
          </div>
        )}

        {/* Latest Predictions Section */}
        <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-white/10">
          <div className="text-[10px] sm:text-xs text-zinc-500 mb-2">Latest Predictions</div>
          <LatestPredictions asset={asset} market={market} />
        </div>
      </div>
    </div>
  );
}
