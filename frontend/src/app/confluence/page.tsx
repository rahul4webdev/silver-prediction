'use client';

import { useState, useEffect } from 'react';
import {
  getConfluence,
  getCorrelations,
  getMacroData,
  ConfluenceData,
  CorrelationData,
  MacroData,
} from '@/lib/api';

export default function ConfluencePage() {
  const [confluence, setConfluence] = useState<ConfluenceData | null>(null);
  const [correlations, setCorrelations] = useState<CorrelationData | null>(null);
  const [macroData, setMacroData] = useState<MacroData | null>(null);
  const [market, setMarket] = useState<'mcx' | 'comex'>('mcx');
  const [lookbackDays, setLookbackDays] = useState(30);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, [market, lookbackDays]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [confData, corrData, mData] = await Promise.all([
        getConfluence('silver', market),
        getCorrelations(lookbackDays),
        getMacroData('silver'),
      ]);
      setConfluence(confData);
      setCorrelations(corrData);
      setMacroData(mData);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
    setLoading(false);
  };

  const getSignalColor = (quality: string) => {
    switch (quality) {
      case 'strong': return 'text-green-400 bg-green-500/20';
      case 'moderate': return 'text-yellow-400 bg-yellow-500/20';
      case 'weak': return 'text-orange-400 bg-orange-500/20';
      default: return 'text-zinc-400 bg-zinc-500/20';
    }
  };

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      default: return 'text-zinc-400';
    }
  };

  const getCorrelationColor = (corr: number) => {
    if (corr > 0.7) return 'text-green-400';
    if (corr > 0.3) return 'text-cyan-400';
    if (corr > -0.3) return 'text-zinc-400';
    if (corr > -0.7) return 'text-orange-400';
    return 'text-red-400';
  };

  const formatCorrelation = (corr: number) => {
    return (corr >= 0 ? '+' : '') + corr.toFixed(2);
  };

  const correlationBarWidth = (corr: number) => {
    return `${Math.abs(corr) * 100}%`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Confluence & Correlations</h1>
          <p className="text-zinc-400 text-sm mt-1">
            Multi-timeframe signals and market correlations
          </p>
        </div>
        <div className="flex items-center gap-4">
          <select
            value={market}
            onChange={(e) => setMarket(e.target.value as 'mcx' | 'comex')}
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
          >
            <option value="mcx">MCX</option>
            <option value="comex">COMEX</option>
          </select>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <div className="glass-card p-8 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mx-auto"></div>
          <p className="text-zinc-400 mt-4">Loading confluence data...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Confluence Signal */}
          <div className="glass-card p-6 space-y-6">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              Multi-Timeframe Confluence
            </h2>

            {confluence ? (
              <>
                {/* Signal Strength */}
                <div className="text-center py-6 border-b border-zinc-700">
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg ${getSignalColor(confluence.signal_quality)}`}>
                    <span className="text-lg font-bold uppercase">{confluence.signal_quality}</span>
                    <span>Signal</span>
                  </div>
                  <div className={`mt-4 text-4xl font-bold ${getDirectionColor(confluence.direction)}`}>
                    {confluence.direction.toUpperCase()}
                  </div>
                  {confluence.confluence_detected && (
                    <p className="text-zinc-400 mt-2">
                      {confluence.aligned_intervals.length} of {confluence.total_intervals} timeframes aligned
                    </p>
                  )}
                </div>

                {/* Strength Meter */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-zinc-400">Signal Strength</span>
                    <span className="text-white font-mono">{(confluence.strength * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        confluence.strength >= 0.8 ? 'bg-green-500' :
                        confluence.strength >= 0.6 ? 'bg-yellow-500' :
                        confluence.strength >= 0.4 ? 'bg-orange-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${confluence.strength * 100}%` }}
                    />
                  </div>
                </div>

                {/* Timeframe Breakdown */}
                <div className="space-y-3">
                  <h3 className="text-sm font-medium text-zinc-400">Timeframe Signals</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(confluence.predictions).map(([interval, pred]) => (
                      <div
                        key={interval}
                        className={`p-3 rounded-lg border ${
                          pred && confluence.aligned_intervals.includes(interval)
                            ? 'border-cyan-500/50 bg-cyan-500/10'
                            : 'border-zinc-700 bg-zinc-800/50'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-zinc-300 font-medium">{interval}</span>
                          {pred ? (
                            <span className={`text-sm ${getDirectionColor(pred.direction)}`}>
                              {pred.direction}
                            </span>
                          ) : (
                            <span className="text-zinc-500 text-sm">No data</span>
                          )}
                        </div>
                        {pred && (
                          <div className="text-xs text-zinc-500 mt-1">
                            Confidence: {(pred.confidence * 100).toFixed(0)}%
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommendation */}
                <div className="p-4 bg-zinc-800/50 rounded-lg">
                  <h3 className="text-sm font-medium text-zinc-400 mb-2">Recommendation</h3>
                  <p className="text-white text-sm">{confluence.recommendation}</p>
                </div>
              </>
            ) : (
              <div className="text-center py-8 text-zinc-500">
                No confluence data available
              </div>
            )}
          </div>

          {/* Correlations */}
          <div className="glass-card p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-400"></span>
                Asset Correlations
              </h2>
              <select
                value={lookbackDays}
                onChange={(e) => setLookbackDays(parseInt(e.target.value))}
                className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-1 text-sm text-white"
              >
                <option value="7">7 days</option>
                <option value="14">14 days</option>
                <option value="30">30 days</option>
                <option value="60">60 days</option>
                <option value="90">90 days</option>
              </select>
            </div>

            {correlations ? (
              <>
                {/* Market Regime */}
                <div className="p-4 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-lg border border-purple-500/20">
                  <div className="text-sm text-zinc-400">Market Regime</div>
                  <div className="text-lg font-bold text-white mt-1 capitalize">
                    {correlations.market_regime.type.replace(/_/g, ' ')}
                  </div>
                  <p className="text-sm text-zinc-400 mt-2">
                    {correlations.market_regime.description}
                  </p>
                </div>

                {/* Correlation Bars */}
                <div className="space-y-4">
                  {Object.entries(correlations.correlations).map(([asset, data]) => {
                    if ('error' in data) return null;
                    const corr = data.correlation;
                    return (
                      <div key={asset} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-zinc-300 capitalize">{asset}</span>
                          <div className="flex items-center gap-2">
                            <span className={getCorrelationColor(corr)}>
                              {formatCorrelation(corr)}
                            </span>
                            <span className={`text-xs px-2 py-0.5 rounded ${
                              data.status === 'normal' ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
                            }`}>
                              {data.status}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center h-2">
                          <div className="flex-1 h-full bg-zinc-800 rounded-l-full overflow-hidden">
                            {corr < 0 && (
                              <div
                                className="h-full bg-red-500 float-right rounded-l-full"
                                style={{ width: `${Math.abs(corr) * 100}%` }}
                              />
                            )}
                          </div>
                          <div className="w-px h-4 bg-zinc-600" />
                          <div className="flex-1 h-full bg-zinc-800 rounded-r-full overflow-hidden">
                            {corr > 0 && (
                              <div
                                className="h-full bg-green-500 rounded-r-full"
                                style={{ width: `${corr * 100}%` }}
                              />
                            )}
                          </div>
                        </div>
                        <div className="flex justify-between text-xs text-zinc-500">
                          <span>-1 (Inverse)</span>
                          <span>0</span>
                          <span>+1 (Positive)</span>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="text-xs text-zinc-500 text-center">
                  Based on {lookbackDays}-day rolling correlation
                </div>
              </>
            ) : (
              <div className="text-center py-8 text-zinc-500">
                No correlation data available
              </div>
            )}
          </div>

          {/* Macro Data */}
          <div className="glass-card p-6 space-y-6 lg:col-span-2">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400"></span>
              Macro Indicators
            </h2>

            {macroData ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* DXY */}
                <div className="p-4 bg-zinc-800/50 rounded-lg">
                  <div className="text-sm text-zinc-400">DXY (Dollar Index)</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {macroData.dxy?.value?.toFixed(2) || 'N/A'}
                  </div>
                  {macroData.dxy?.change_1d !== undefined && (
                    <div className={`text-sm mt-1 ${
                      macroData.dxy.change_1d >= 0 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {macroData.dxy.change_1d >= 0 ? '+' : ''}{macroData.dxy.change_1d.toFixed(2)}%
                      <span className="text-zinc-500 ml-1">(1d)</span>
                    </div>
                  )}
                  <div className="text-xs text-zinc-500 mt-1">
                    Inverse correlation with silver
                  </div>
                </div>

                {/* Fear & Greed */}
                <div className="p-4 bg-zinc-800/50 rounded-lg">
                  <div className="text-sm text-zinc-400">Fear & Greed Index</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {macroData.fear_greed?.value || 'N/A'}
                  </div>
                  <div className={`text-sm mt-1 ${
                    macroData.fear_greed?.label === 'Extreme Fear' || macroData.fear_greed?.label === 'Fear'
                      ? 'text-red-400'
                      : macroData.fear_greed?.label === 'Extreme Greed' || macroData.fear_greed?.label === 'Greed'
                      ? 'text-green-400'
                      : 'text-zinc-400'
                  }`}>
                    {macroData.fear_greed?.label || 'Unknown'}
                  </div>
                  <div className="text-xs text-zinc-500 mt-1">
                    Market sentiment gauge
                  </div>
                </div>

                {/* COT Data */}
                <div className="p-4 bg-zinc-800/50 rounded-lg">
                  <div className="text-sm text-zinc-400">COT Net Positions</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {macroData.cot?.net_positions?.toLocaleString() || 'N/A'}
                  </div>
                  {macroData.cot?.change_week !== undefined && (
                    <div className={`text-sm mt-1 ${
                      macroData.cot.change_week >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {macroData.cot.change_week >= 0 ? '+' : ''}{macroData.cot.change_week.toLocaleString()}
                      <span className="text-zinc-500 ml-1">(weekly)</span>
                    </div>
                  )}
                  <div className="text-xs text-zinc-500 mt-1">
                    Speculator positioning
                  </div>
                </div>

                {/* Treasury */}
                <div className="p-4 bg-zinc-800/50 rounded-lg">
                  <div className="text-sm text-zinc-400">US 10Y Yield</div>
                  <div className="text-2xl font-bold text-white mt-1">
                    {macroData.treasury?.us_10y?.toFixed(2) || 'N/A'}%
                  </div>
                  {macroData.treasury?.spread_10y_2y !== undefined && (
                    <div className={`text-sm mt-1 ${
                      macroData.treasury.spread_10y_2y >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      Spread: {macroData.treasury.spread_10y_2y.toFixed(2)}%
                    </div>
                  )}
                  <div className="text-xs text-zinc-500 mt-1">
                    Real rates impact silver
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-zinc-500">
                No macro data available
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
