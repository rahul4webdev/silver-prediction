'use client';

import { useEffect, useRef } from 'react';
import { createChart, IChartApi, LineData, Time } from 'lightweight-charts';

interface AccuracyTrendChartProps {
  data: { date: string; accuracy: number }[];
}

export default function AccuracyTrendChart({ data }: AccuracyTrendChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

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
      height: 250,
      rightPriceScale: {
        borderColor: '#e0e0e0',
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: '#e0e0e0',
      },
    });

    chartRef.current = chart;

    const lineSeries = chart.addLineSeries({
      color: '#6366f1',
      lineWidth: 2,
      title: 'Accuracy %',
    });

    // Add target line at 55%
    const targetLine = chart.addLineSeries({
      color: '#22c55e',
      lineWidth: 1,
      lineStyle: 2,
      title: 'Target (55%)',
    });

    if (data.length > 0) {
      const lineData: LineData[] = data.map((d) => ({
        time: (new Date(d.date).getTime() / 1000) as Time,
        value: d.accuracy * 100,
      }));
      lineSeries.setData(lineData);

      // Set target line across the entire range
      const targetData: LineData[] = [
        { time: lineData[0].time, value: 55 },
        { time: lineData[lineData.length - 1].time, value: 55 },
      ];
      targetLine.setData(targetData);

      chart.timeScale().fitContent();
    }

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
  }, [data]);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[250px] text-gray-500">
        No trend data available
      </div>
    );
  }

  return <div ref={chartContainerRef} className="w-full" />;
}
