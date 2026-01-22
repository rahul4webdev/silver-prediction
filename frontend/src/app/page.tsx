'use client';

import { useState } from 'react';
import PriceCard from '@/components/PriceCard';
import PredictionCard from '@/components/PredictionCard';
import PriceChart from '@/components/PriceChart';
import AccuracyCard from '@/components/AccuracyCard';
import SentimentCard from '@/components/SentimentCard';
import type { Market, Interval } from '@/lib/types';

const intervals: { value: Interval; label: string }[] = [
  { value: '30m', label: '30 Min' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: 'Daily' },
];

export default function Dashboard() {
  const [selectedMarket, setSelectedMarket] = useState<Market>('mcx');
  const [selectedInterval, setSelectedInterval] = useState<Interval>('1h');

  return (
    <div className="space-y-6">
      {/* Market & Interval Selector */}
      <div className="flex flex-wrap items-center gap-4">
        {/* Market Toggle */}
        <div className="glass-card p-1 flex">
          <button
            onClick={() => setSelectedMarket('mcx')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMarket === 'mcx'
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            MCX (India)
          </button>
          <button
            onClick={() => setSelectedMarket('comex')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMarket === 'comex'
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            COMEX (US)
          </button>
        </div>

        {/* Interval Selector */}
        <div className="glass-card p-1 flex">
          {intervals.map((interval) => (
            <button
              key={interval.value}
              onClick={() => setSelectedInterval(interval.value)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedInterval === interval.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              {interval.label}
            </button>
          ))}
        </div>
      </div>

      {/* Top Row: Price Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PriceCard market="mcx" />
        <PriceCard market="comex" />
      </div>

      {/* Main Chart */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">
            Silver Price Chart - {selectedMarket.toUpperCase()}
          </h2>
          <span className="text-xs text-zinc-500">
            {selectedInterval === '30m' && '30 Minute Candles'}
            {selectedInterval === '1h' && '1 Hour Candles'}
            {selectedInterval === '4h' && '4 Hour Candles'}
            {selectedInterval === '1d' && 'Daily Candles'}
          </span>
        </div>
        <PriceChart market={selectedMarket} interval={selectedInterval} />
      </div>

      {/* Bottom Row: Prediction, Accuracy & Sentiment */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <PredictionCard market={selectedMarket} interval={selectedInterval} />
        <AccuracyCard />
        <SentimentCard asset="silver" />
      </div>

      {/* Quick Stats */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-medium text-zinc-400 mb-4">Market Overview</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Selected Market</div>
            <div className="text-lg font-bold text-white">{selectedMarket.toUpperCase()}</div>
          </div>
          <div className="text-center p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Prediction Interval</div>
            <div className="text-lg font-bold text-white">
              {intervals.find(i => i.value === selectedInterval)?.label}
            </div>
          </div>
          <div className="text-center p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Asset</div>
            <div className="text-lg font-bold text-white">Silver</div>
          </div>
          <div className="text-center p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Data Source</div>
            <div className="text-lg font-bold text-cyan-400">
              {selectedMarket === 'mcx' ? 'Silver Bees ETF' : 'Yahoo Finance'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
