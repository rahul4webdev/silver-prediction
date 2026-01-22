'use client';

import { useEffect, useState } from 'react';
import { getLivePrice } from '@/lib/api';
import { formatCurrency, formatPercent, cn } from '@/lib/utils';
import type { LivePrice } from '@/lib/types';

interface PriceCardProps {
  market: 'mcx' | 'comex';
}

export default function PriceCard({ market }: PriceCardProps) {
  const [price, setPrice] = useState<LivePrice | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPrice() {
      try {
        setLoading(true);
        const data = await getLivePrice('silver', market);
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
    const interval = setInterval(fetchPrice, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [market]);

  const isPositive = (price?.change_percent ?? 0) >= 0;
  const currency = market === 'mcx' ? 'INR' : 'USD';

  if (loading) {
    return (
      <div className="glass-card p-6">
        <div className="skeleton h-4 w-24 rounded mb-4"></div>
        <div className="skeleton h-10 w-40 rounded mb-2"></div>
        <div className="skeleton h-4 w-20 rounded"></div>
      </div>
    );
  }

  if (error || !price) {
    return (
      <div className="glass-card p-6">
        <div className="text-zinc-400 text-sm">{market.toUpperCase()} Silver</div>
        <div className="text-zinc-500 mt-2">Unable to load price</div>
      </div>
    );
  }

  return (
    <div className={cn(
      "glass-card p-6 relative overflow-hidden",
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
        <div className="flex items-center justify-between mb-3">
          <span className="text-zinc-400 text-sm font-medium">{market.toUpperCase()} Silver</span>
          <span className="text-xs text-zinc-500 bg-white/5 px-2 py-1 rounded">
            {price.source?.replace('_', ' ')}
          </span>
        </div>

        <div className="text-3xl font-bold text-white mb-2">
          {formatCurrency(price.price, currency)}
        </div>

        <div className={cn(
          "text-sm font-medium",
          isPositive ? "text-bullish" : "text-bearish"
        )}>
          {price.change_percent !== undefined && formatPercent(price.change_percent)}
        </div>

        {market === 'mcx' && (
          <div className="text-xs text-zinc-500 mt-2">
            Price per kg
          </div>
        )}
      </div>
    </div>
  );
}
