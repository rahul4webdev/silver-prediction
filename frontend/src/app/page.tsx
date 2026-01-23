'use client';

import { useState } from 'react';
import PriceCard from '@/components/PriceCard';
import PredictionCard from '@/components/PredictionCard';
import PriceChart from '@/components/PriceChart';
import AccuracyCard from '@/components/AccuracyCard';
import SentimentCard from '@/components/SentimentCard';
import type { Market, Interval } from '@/lib/types';

const intervals: { value: Interval; label: string; shortLabel: string }[] = [
  { value: '30m', label: '30 Min', shortLabel: '30m' },
  { value: '1h', label: '1 Hour', shortLabel: '1h' },
  { value: '4h', label: '4 Hours', shortLabel: '4h' },
  { value: '1d', label: 'Daily', shortLabel: '1d' },
];

export default function Dashboard() {
  const [selectedMarket, setSelectedMarket] = useState<Market>('mcx');
  const [selectedInterval, setSelectedInterval] = useState<Interval>('1h');

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Market & Interval Selector */}
      <div className="flex flex-col sm:flex-row gap-2 sm:gap-4">
        {/* Market Toggle */}
        <div className="glass-card p-1 flex flex-1 sm:flex-initial">
          <button
            onClick={() => setSelectedMarket('mcx')}
            className={`flex-1 sm:flex-initial px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMarket === 'mcx'
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            <span className="hidden sm:inline">MCX (India)</span>
            <span className="sm:hidden">MCX</span>
          </button>
          <button
            onClick={() => setSelectedMarket('comex')}
            className={`flex-1 sm:flex-initial px-3 sm:px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedMarket === 'comex'
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-zinc-400 hover:text-white'
            }`}
          >
            <span className="hidden sm:inline">COMEX (US)</span>
            <span className="sm:hidden">COMEX</span>
          </button>
        </div>

        {/* Interval Selector */}
        <div className="glass-card p-1 flex flex-1 sm:flex-initial">
          {intervals.map((interval) => (
            <button
              key={interval.value}
              onClick={() => setSelectedInterval(interval.value)}
              className={`flex-1 sm:flex-initial px-2 sm:px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedInterval === interval.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <span className="hidden sm:inline">{interval.label}</span>
              <span className="sm:hidden">{interval.shortLabel}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Top Row: Price Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
        <PriceCard market="mcx" />
        <PriceCard market="comex" />
      </div>

      {/* Main Chart */}
      <div className="glass-card p-3 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
          <h2 className="text-base sm:text-lg font-semibold text-white">
            Silver - {selectedMarket.toUpperCase()}
          </h2>
          <span className="text-xs text-zinc-500">
            {selectedInterval === '30m' && '30 Minute Candles'}
            {selectedInterval === '1h' && '1 Hour Candles'}
            {selectedInterval === '4h' && '4 Hour Candles'}
            {selectedInterval === '1d' && 'Daily Candles'}
          </span>
        </div>
        <div className="h-[300px] sm:h-auto">
          <PriceChart market={selectedMarket} interval={selectedInterval} />
        </div>
      </div>

      {/* Bottom Row: Prediction, Accuracy & Sentiment */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
        <PredictionCard market={selectedMarket} interval={selectedInterval} />
        <AccuracyCard />
        <SentimentCard asset="silver" />
      </div>

      {/* Quick Stats - Hidden on mobile to save space */}
      <div className="glass-card p-4 sm:p-6 hidden sm:block">
        <h3 className="text-sm font-medium text-zinc-400 mb-4">Market Overview</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          <div className="text-center p-3 sm:p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Selected Market</div>
            <div className="text-base sm:text-lg font-bold text-white">{selectedMarket.toUpperCase()}</div>
          </div>
          <div className="text-center p-3 sm:p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Prediction Interval</div>
            <div className="text-base sm:text-lg font-bold text-white">
              {intervals.find(i => i.value === selectedInterval)?.label}
            </div>
          </div>
          <div className="text-center p-3 sm:p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Asset</div>
            <div className="text-base sm:text-lg font-bold text-white">Silver</div>
          </div>
          <div className="text-center p-3 sm:p-4 bg-white/5 rounded-xl">
            <div className="text-xs text-zinc-500 mb-1">Data Source</div>
            <div className="text-base sm:text-lg font-bold text-cyan-400">
              {selectedMarket === 'mcx' ? 'Silver Bees ETF' : 'Yahoo Finance'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
