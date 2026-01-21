'use client';

import { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData, Time } from 'lightweight-charts';
import type { PriceCandle, Prediction } from '@/lib/types';

interface PredictionChartProps {
  priceData: PriceCandle[];
  prediction: Prediction | null;
  market: 'mcx' | 'comex';
  interval: string;
}

export default function PredictionChart({
  priceData,
  prediction,
  market,
  interval,
}: PredictionChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const predictionLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ci50UpperRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ci50LowerRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ci80UpperRef = useRef<ISeriesApi<'Line'> | null>(null);
  const ci80LowerRef = useRef<ISeriesApi<'Line'> | null>(null);
  const [showCIBands, setShowCIBands] = useState(true);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#e0e0e0',
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: '#e0e0e0',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    // Create candlestick series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });
    candleSeriesRef.current = candleSeries;

    // Create prediction line
    const predictionLine = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 2,
      lineStyle: 2, // Dashed
      title: 'Prediction',
    });
    predictionLineRef.current = predictionLine;

    // Create CI bands
    const ci80Upper = chart.addLineSeries({
      color: 'rgba(139, 92, 246, 0.3)',
      lineWidth: 1,
      lineStyle: 1,
      title: '80% CI Upper',
    });
    ci80UpperRef.current = ci80Upper;

    const ci80Lower = chart.addLineSeries({
      color: 'rgba(139, 92, 246, 0.3)',
      lineWidth: 1,
      lineStyle: 1,
      title: '80% CI Lower',
    });
    ci80LowerRef.current = ci80Lower;

    const ci50Upper = chart.addLineSeries({
      color: 'rgba(139, 92, 246, 0.5)',
      lineWidth: 1,
      lineStyle: 1,
      title: '50% CI Upper',
    });
    ci50UpperRef.current = ci50Upper;

    const ci50Lower = chart.addLineSeries({
      color: 'rgba(139, 92, 246, 0.5)',
      lineWidth: 1,
      lineStyle: 1,
      title: '50% CI Lower',
    });
    ci50LowerRef.current = ci50Lower;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // Update price data
  useEffect(() => {
    if (!candleSeriesRef.current || priceData.length === 0) return;

    const candleData: CandlestickData[] = priceData.map((candle) => ({
      time: (new Date(candle.timestamp).getTime() / 1000) as Time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));

    candleSeriesRef.current.setData(candleData);

    // Fit content
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, [priceData]);

  // Update prediction visualization
  useEffect(() => {
    if (!prediction || !predictionLineRef.current || priceData.length === 0) {
      // Clear prediction lines if no prediction
      predictionLineRef.current?.setData([]);
      ci50UpperRef.current?.setData([]);
      ci50LowerRef.current?.setData([]);
      ci80UpperRef.current?.setData([]);
      ci80LowerRef.current?.setData([]);
      return;
    }

    const lastCandle = priceData[priceData.length - 1];
    const lastTime = new Date(lastCandle.timestamp).getTime() / 1000;
    const targetTime = new Date(prediction.target_time).getTime() / 1000;

    // Prediction line from current price to predicted price
    const predictionData: LineData[] = [
      { time: lastTime as Time, value: prediction.current_price },
      { time: targetTime as Time, value: prediction.predicted_price },
    ];
    predictionLineRef.current.setData(predictionData);

    if (showCIBands && prediction.ci_80_upper && prediction.ci_80_lower) {
      // 80% CI bands
      const ci80UpperData: LineData[] = [
        { time: lastTime as Time, value: prediction.current_price },
        { time: targetTime as Time, value: prediction.ci_80_upper },
      ];
      const ci80LowerData: LineData[] = [
        { time: lastTime as Time, value: prediction.current_price },
        { time: targetTime as Time, value: prediction.ci_80_lower },
      ];
      ci80UpperRef.current?.setData(ci80UpperData);
      ci80LowerRef.current?.setData(ci80LowerData);

      // 50% CI bands
      if (prediction.ci_50_upper && prediction.ci_50_lower) {
        const ci50UpperData: LineData[] = [
          { time: lastTime as Time, value: prediction.current_price },
          { time: targetTime as Time, value: prediction.ci_50_upper },
        ];
        const ci50LowerData: LineData[] = [
          { time: lastTime as Time, value: prediction.current_price },
          { time: targetTime as Time, value: prediction.ci_50_lower },
        ];
        ci50UpperRef.current?.setData(ci50UpperData);
        ci50LowerRef.current?.setData(ci50LowerData);
      }
    } else {
      ci50UpperRef.current?.setData([]);
      ci50LowerRef.current?.setData([]);
      ci80UpperRef.current?.setData([]);
      ci80LowerRef.current?.setData([]);
    }
  }, [prediction, priceData, showCIBands]);

  return (
    <div className="relative">
      {/* Chart Controls */}
      <div className="absolute top-4 left-4 z-10 flex items-center gap-4">
        <div className="bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm border border-gray-200">
          <span className="text-sm font-medium text-gray-700">
            Silver {market.toUpperCase()} • {interval}
          </span>
        </div>
        <label className="flex items-center gap-2 bg-white/90 backdrop-blur px-3 py-1.5 rounded-lg shadow-sm border border-gray-200 cursor-pointer">
          <input
            type="checkbox"
            checked={showCIBands}
            onChange={(e) => setShowCIBands(e.target.checked)}
            className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
          />
          <span className="text-sm text-gray-700">Show CI Bands</span>
        </label>
      </div>

      {/* Legend */}
      <div className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur px-3 py-2 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span>Bullish</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Bearish</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-purple-500"></div>
            <span>Prediction</span>
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div ref={chartContainerRef} className="w-full" />

      {/* Prediction Marker */}
      {prediction && (
        <div className="absolute bottom-4 left-4 z-10 bg-white/90 backdrop-blur px-4 py-3 rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center gap-4">
            <div>
              <p className="text-xs text-gray-500">Predicted Direction</p>
              <span className={`badge-${prediction.predicted_direction} mt-1`}>
                {prediction.predicted_direction.toUpperCase()}
              </span>
            </div>
            <div className="border-l border-gray-200 pl-4">
              <p className="text-xs text-gray-500">Confidence</p>
              <p className="text-lg font-semibold">
                {(prediction.direction_confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div className="border-l border-gray-200 pl-4">
              <p className="text-xs text-gray-500">Target Time</p>
              <p className="text-sm font-medium">
                {new Date(prediction.target_time).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
