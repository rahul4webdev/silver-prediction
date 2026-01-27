'use client';

import { useEffect, useState } from 'react';
import { getSystemStatus, type SystemStatus, type ServiceStatus, type ModelStatus } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function StatusPage() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchStatus = async () => {
    try {
      const data = await getSystemStatus();
      setStatus(data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to fetch status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'running':
      case 'authenticated':
      case 'open':
      case 'available':
        return 'text-green-400 bg-green-500/20';
      case 'operational':
        return 'text-blue-400 bg-blue-500/20';
      case 'degraded':
      case 'needs_auth':
      case 'stopped':
        return 'text-yellow-400 bg-yellow-500/20';
      case 'unhealthy':
      case 'error':
      case 'closed':
        return 'text-red-400 bg-red-500/20';
      default:
        return 'text-zinc-400 bg-zinc-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'running':
      case 'authenticated':
      case 'available':
        return '✓';
      case 'operational':
        return '◐';
      case 'degraded':
      case 'needs_auth':
      case 'stopped':
        return '!';
      case 'unhealthy':
      case 'error':
        return '✗';
      default:
        return '?';
    }
  };

  const ServiceCard = ({ name, service }: { name: string; service: ServiceStatus }) => {
    const memoryUsed = service.memory_used as string | undefined;
    const userName = service.user_name as string | undefined;
    const tickCount = service.tick_count as number | undefined;
    const reason = service.reason as string | undefined;
    const reauthUrl = service.reauth_url as string | undefined;

    return (
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-medium text-white capitalize">{name.replace(/_/g, ' ')}</h3>
          <span className={cn('px-2 py-0.5 rounded-full text-xs font-medium', getStatusColor(service.status))}>
            {getStatusIcon(service.status)} {service.status}
          </span>
        </div>
        {service.error ? (
          <p className="text-xs text-red-400 mt-1 truncate" title={service.error}>
            {service.error}
          </p>
        ) : null}
        {reason && service.status === 'needs_auth' ? (
          <p className="text-xs text-yellow-400 mt-1">{reason === 'token_expired' ? 'Token expired' : reason === 'no_token' ? 'No token set' : reason}</p>
        ) : null}
        {reauthUrl && service.status === 'needs_auth' ? (
          <a
            href={`${process.env.NEXT_PUBLIC_API_URL || 'https://predictionapi.gahfaudio.in'}${reauthUrl}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-cyan-400 hover:text-cyan-300 mt-1 block"
          >
            Click to re-authenticate →
          </a>
        ) : null}
        {memoryUsed ? (
          <p className="text-xs text-zinc-500 mt-1">Memory: {memoryUsed}</p>
        ) : null}
        {userName ? (
          <p className="text-xs text-zinc-500 mt-1">User: {userName}</p>
        ) : null}
        {typeof tickCount === 'number' ? (
          <p className="text-xs text-zinc-500 mt-1">Ticks: {tickCount.toLocaleString()}</p>
        ) : null}
      </div>
    );
  };

  const ModelCard = ({ interval, model }: { interval: string; model: ModelStatus }) => {
    const intervalLabels: Record<string, string> = {
      '30m': '30 Minutes',
      '1h': '1 Hour',
      '4h': '4 Hours',
      '1d': 'Daily',
    };

    return (
      <div className="glass-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-white">{intervalLabels[interval] || interval}</h3>
          <span className={cn(
            'px-2 py-0.5 rounded-full text-xs font-medium',
            model.is_trained ? 'text-green-400 bg-green-500/20' : 'text-yellow-400 bg-yellow-500/20'
          )}>
            {model.is_trained ? '✓ Trained' : '! Untrained'}
          </span>
        </div>
        {model.models && (
          <div className="space-y-1.5">
            {Object.entries(model.models).map(([name, m]) => (
              <div key={name} className="flex items-center justify-between text-xs">
                <span className="text-zinc-400 capitalize">{name}</span>
                <div className="flex items-center gap-2">
                  <span className="text-zinc-500">
                    {(m.weight * 100).toFixed(0)}%
                  </span>
                  <span className={m.is_trained ? 'text-green-400' : 'text-zinc-600'}>
                    {m.is_trained ? '●' : '○'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
        {model.error && (
          <p className="text-xs text-red-400 mt-2 truncate" title={model.error}>
            {model.error}
          </p>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-zinc-400 mt-4">Loading system status...</p>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="glass-card p-8 text-center">
        <p className="text-red-400">Failed to load system status</p>
        <button
          onClick={fetchStatus}
          className="mt-4 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">System Status</h1>
            <p className="text-zinc-400">Monitor all services, models, and scheduled tasks.</p>
          </div>
          <div className="text-right">
            <span className={cn(
              'inline-flex items-center gap-2 px-4 py-2 rounded-lg text-lg font-medium',
              getStatusColor(status.overall_health)
            )}>
              {getStatusIcon(status.overall_health)} {status.overall_health.toUpperCase()}
            </span>
            {status.overall_health === 'operational' && (
              <p className="text-xs text-blue-400 mt-1">Using Yahoo Finance (Upstox not connected)</p>
            )}
            {lastUpdated && (
              <p className="text-xs text-zinc-500 mt-2">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Market Status */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Market Status</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className={cn(
              'text-2xl font-bold mb-1',
              status.market.is_trading_hours ? 'text-green-400' : 'text-red-400'
            )}>
              {status.market.market_status.toUpperCase()}
            </div>
            <div className="text-xs text-zinc-500">Market</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-medium text-white mb-1">
              {status.market.current_time_ist}
            </div>
            <div className="text-xs text-zinc-500">Current Time</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-medium text-white mb-1">
              {status.market.hours.open} - {status.market.hours.close}
            </div>
            <div className="text-xs text-zinc-500">Trading Hours</div>
          </div>
          <div className="text-center">
            <div className={cn(
              'text-lg font-medium mb-1',
              status.market.is_weekend ? 'text-yellow-400' : 'text-green-400'
            )}>
              {status.market.is_weekend ? 'Weekend' : 'Weekday'}
            </div>
            <div className="text-xs text-zinc-500">Day Type</div>
          </div>
        </div>
      </div>

      {/* Services */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Services</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {Object.entries(status.services).map(([name, service]) => (
            <ServiceCard key={name} name={name} service={service} />
          ))}
        </div>
      </div>

      {/* ML Models */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">ML Models</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(status.models).map(([interval, model]) => (
            <ModelCard key={interval} interval={interval} model={model} />
          ))}
        </div>
      </div>

      {/* Database Stats */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Database Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-cyan-400">
              {status.database.predictions.total.toLocaleString()}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Total Predictions</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400">
              {status.database.predictions.verified.toLocaleString()}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Verified</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-yellow-400">
              {status.database.predictions.pending.toLocaleString()}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Pending</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-white">
              {status.database.predictions.today.toLocaleString()}
            </div>
            <div className="text-xs text-zinc-500 mt-1">Today</div>
          </div>
        </div>
      </div>

      {/* Scheduler Jobs */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Scheduled Jobs</h2>
          <span className={cn(
            'px-2 py-0.5 rounded-full text-xs font-medium',
            getStatusColor(status.scheduler.status)
          )}>
            {status.scheduler.is_running ? '● Running' : '○ Stopped'}
          </span>
        </div>
        {status.scheduler.jobs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs text-zinc-500 font-medium py-2 px-3">Job Name</th>
                  <th className="text-left text-xs text-zinc-500 font-medium py-2 px-3">Job ID</th>
                  <th className="text-right text-xs text-zinc-500 font-medium py-2 px-3">Next Run</th>
                </tr>
              </thead>
              <tbody>
                {status.scheduler.jobs.map((job) => (
                  <tr key={job.id} className="border-b border-white/5 hover:bg-white/5">
                    <td className="py-2 px-3 text-sm text-white">{job.name}</td>
                    <td className="py-2 px-3 text-sm text-zinc-400 font-mono">{job.id}</td>
                    <td className="py-2 px-3 text-sm text-zinc-400 text-right">
                      {job.next_run ? new Date(job.next_run).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-zinc-500 text-center py-4">No scheduled jobs</p>
        )}
      </div>

      {/* Environment Info */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-zinc-500">Environment: <span className="text-white">{status.environment}</span></span>
          <span className="text-zinc-500">Timestamp: <span className="text-white">{new Date(status.timestamp).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} IST</span></span>
        </div>
      </div>
    </div>
  );
}
