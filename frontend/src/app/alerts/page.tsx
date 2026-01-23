'use client';

import { useState, useEffect } from 'react';
import {
  getAlerts,
  createAlert,
  deleteAlert,
  getTrades,
  createTrade,
  closeTrade,
  getTradePerformance,
  getLivePrice,
  PriceAlert,
  Trade,
  TradePerformance,
} from '@/lib/api';

export default function AlertsPage() {
  const [activeTab, setActiveTab] = useState<'alerts' | 'trades'>('alerts');
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [performance, setPerformance] = useState<TradePerformance | null>(null);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [showCreateAlert, setShowCreateAlert] = useState(false);
  const [showCreateTrade, setShowCreateTrade] = useState(false);

  // Alert form state
  const [alertForm, setAlertForm] = useState({
    asset: 'silver',
    market: 'mcx',
    alert_type: 'above' as 'above' | 'below',
    target_price: '',
    notes: '',
  });

  // Trade form state
  const [tradeForm, setTradeForm] = useState({
    asset: 'silver',
    market: 'mcx',
    trade_type: 'long' as 'long' | 'short',
    entry_price: '',
    quantity: '',
    notes: '',
  });

  // Close trade state
  const [closingTrade, setClosingTrade] = useState<string | null>(null);
  const [exitPrice, setExitPrice] = useState('');

  useEffect(() => {
    fetchData();
    // Refresh price every 30 seconds
    const interval = setInterval(() => {
      fetchCurrentPrice();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [alertsData, tradesData, perfData, priceData] = await Promise.all([
        getAlerts('silver'),
        getTrades('silver'),
        getTradePerformance('silver'),
        getLivePrice('silver', 'mcx'),
      ]);
      setAlerts(alertsData.alerts);
      setTrades(tradesData.trades);
      setPerformance(perfData);
      if (priceData?.price) {
        setCurrentPrice(priceData.price);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
    }
    setLoading(false);
  };

  const fetchCurrentPrice = async () => {
    try {
      const priceData = await getLivePrice('silver', 'mcx');
      if (priceData?.price) {
        setCurrentPrice(priceData.price);
      }
    } catch (error) {
      console.error('Error fetching price:', error);
    }
  };

  const handleCreateAlert = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await createAlert(
      alertForm.asset,
      alertForm.market,
      alertForm.alert_type,
      parseFloat(alertForm.target_price),
      alertForm.notes || undefined
    );
    if (result) {
      setShowCreateAlert(false);
      setAlertForm({ ...alertForm, target_price: '', notes: '' });
      fetchData();
    }
  };

  const handleDeleteAlert = async (alertId: string) => {
    const success = await deleteAlert(alertId);
    if (success) {
      fetchData();
    }
  };

  const handleCreateTrade = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await createTrade({
      asset: tradeForm.asset,
      market: tradeForm.market,
      trade_type: tradeForm.trade_type,
      entry_price: parseFloat(tradeForm.entry_price),
      quantity: parseFloat(tradeForm.quantity),
      notes: tradeForm.notes || undefined,
    });
    if (result) {
      setShowCreateTrade(false);
      setTradeForm({ ...tradeForm, entry_price: '', quantity: '', notes: '' });
      fetchData();
    }
  };

  const handleCloseTrade = async (tradeId: string) => {
    if (!exitPrice) return;
    const result = await closeTrade(tradeId, parseFloat(exitPrice));
    if (result) {
      setClosingTrade(null);
      setExitPrice('');
      fetchData();
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Alerts & Trade Journal</h1>
          <p className="text-zinc-400 text-sm mt-1">
            Manage price alerts and track your trades
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="glass-card px-4 py-2">
            <span className="text-zinc-400 text-sm">Current Price:</span>
            <span className="text-white font-mono ml-2">{formatCurrency(currentPrice)}</span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-zinc-700 pb-2">
        <button
          onClick={() => setActiveTab('alerts')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            activeTab === 'alerts'
              ? 'bg-cyan-500/20 text-cyan-400'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
        >
          Price Alerts
        </button>
        <button
          onClick={() => setActiveTab('trades')}
          className={`px-4 py-2 rounded-lg transition-colors ${
            activeTab === 'trades'
              ? 'bg-cyan-500/20 text-cyan-400'
              : 'text-zinc-400 hover:text-white hover:bg-white/5'
          }`}
        >
          Trade Journal
        </button>
      </div>

      {/* Alerts Tab */}
      {activeTab === 'alerts' && (
        <div className="space-y-4">
          <div className="flex justify-end">
            <button
              onClick={() => setShowCreateAlert(true)}
              className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
            >
              + New Alert
            </button>
          </div>

          {/* Create Alert Modal */}
          {showCreateAlert && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="glass-card p-6 w-full max-w-md mx-4">
                <h2 className="text-lg font-bold text-white mb-4">Create Price Alert</h2>
                <form onSubmit={handleCreateAlert} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Asset</label>
                      <select
                        value={alertForm.asset}
                        onChange={(e) => setAlertForm({ ...alertForm, asset: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      >
                        <option value="silver">Silver</option>
                        <option value="gold">Gold</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Market</label>
                      <select
                        value={alertForm.market}
                        onChange={(e) => setAlertForm({ ...alertForm, market: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      >
                        <option value="mcx">MCX</option>
                        <option value="comex">COMEX</option>
                      </select>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Alert Type</label>
                      <select
                        value={alertForm.alert_type}
                        onChange={(e) => setAlertForm({ ...alertForm, alert_type: e.target.value as 'above' | 'below' })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      >
                        <option value="above">Price Above</option>
                        <option value="below">Price Below</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Target Price</label>
                      <input
                        type="number"
                        step="0.01"
                        value={alertForm.target_price}
                        onChange={(e) => setAlertForm({ ...alertForm, target_price: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                        placeholder="Enter price"
                        required
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-400 mb-1">Notes (optional)</label>
                    <textarea
                      value={alertForm.notes}
                      onChange={(e) => setAlertForm({ ...alertForm, notes: e.target.value })}
                      className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      rows={2}
                    />
                  </div>
                  <div className="flex gap-3 justify-end">
                    <button
                      type="button"
                      onClick={() => setShowCreateAlert(false)}
                      className="px-4 py-2 text-zinc-400 hover:text-white transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
                    >
                      Create Alert
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Alerts List */}
          {loading ? (
            <div className="glass-card p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mx-auto"></div>
            </div>
          ) : alerts.length === 0 ? (
            <div className="glass-card p-8 text-center">
              <p className="text-zinc-400">No price alerts set. Create one to get notified!</p>
            </div>
          ) : (
            <div className="grid gap-4">
              {alerts.map((alert) => (
                <div key={alert.id} className="glass-card p-4 flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      alert.alert_type === 'above' ? 'bg-green-500/20' : 'bg-red-500/20'
                    }`}>
                      {alert.alert_type === 'above' ? (
                        <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                        </svg>
                      ) : (
                        <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-white font-medium">{alert.asset.toUpperCase()}</span>
                        <span className="text-zinc-500">({alert.market.toUpperCase()})</span>
                        <span className={`px-2 py-0.5 rounded text-xs ${
                          alert.status === 'active' ? 'bg-cyan-500/20 text-cyan-400' :
                          alert.status === 'triggered' ? 'bg-green-500/20 text-green-400' :
                          'bg-zinc-500/20 text-zinc-400'
                        }`}>
                          {alert.status}
                        </span>
                      </div>
                      <p className="text-sm text-zinc-400">
                        Alert when price {alert.alert_type} {formatCurrency(alert.target_price)}
                      </p>
                      {alert.notes && (
                        <p className="text-xs text-zinc-500 mt-1">{alert.notes}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {alert.status === 'active' && (
                      <button
                        onClick={() => handleDeleteAlert(alert.id)}
                        className="p-2 text-zinc-400 hover:text-red-400 transition-colors"
                        title="Delete alert"
                      >
                        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Trades Tab */}
      {activeTab === 'trades' && (
        <div className="space-y-4">
          {/* Performance Summary */}
          {performance && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="glass-card p-4">
                <p className="text-zinc-400 text-sm">Win Rate</p>
                <p className={`text-2xl font-bold ${performance.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}`}>
                  {performance.win_rate.toFixed(1)}%
                </p>
              </div>
              <div className="glass-card p-4">
                <p className="text-zinc-400 text-sm">Total P&L</p>
                <p className={`text-2xl font-bold ${performance.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(performance.total_pnl)}
                </p>
              </div>
              <div className="glass-card p-4">
                <p className="text-zinc-400 text-sm">Total Trades</p>
                <p className="text-2xl font-bold text-white">{performance.total_trades}</p>
              </div>
              <div className="glass-card p-4">
                <p className="text-zinc-400 text-sm">Prediction Accuracy</p>
                <p className="text-2xl font-bold text-cyan-400">
                  {performance.followed_prediction_accuracy.toFixed(1)}%
                </p>
              </div>
            </div>
          )}

          <div className="flex justify-end">
            <button
              onClick={() => setShowCreateTrade(true)}
              className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
            >
              + Log Trade
            </button>
          </div>

          {/* Create Trade Modal */}
          {showCreateTrade && (
            <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
              <div className="glass-card p-6 w-full max-w-md mx-4">
                <h2 className="text-lg font-bold text-white mb-4">Log New Trade</h2>
                <form onSubmit={handleCreateTrade} className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Asset</label>
                      <select
                        value={tradeForm.asset}
                        onChange={(e) => setTradeForm({ ...tradeForm, asset: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      >
                        <option value="silver">Silver</option>
                        <option value="gold">Gold</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Market</label>
                      <select
                        value={tradeForm.market}
                        onChange={(e) => setTradeForm({ ...tradeForm, market: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      >
                        <option value="mcx">MCX</option>
                        <option value="comex">COMEX</option>
                      </select>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Type</label>
                      <select
                        value={tradeForm.trade_type}
                        onChange={(e) => setTradeForm({ ...tradeForm, trade_type: e.target.value as 'long' | 'short' })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      >
                        <option value="long">Long</option>
                        <option value="short">Short</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Entry Price</label>
                      <input
                        type="number"
                        step="0.01"
                        value={tradeForm.entry_price}
                        onChange={(e) => setTradeForm({ ...tradeForm, entry_price: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                        required
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-400 mb-1">Quantity</label>
                      <input
                        type="number"
                        step="0.01"
                        value={tradeForm.quantity}
                        onChange={(e) => setTradeForm({ ...tradeForm, quantity: e.target.value })}
                        className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                        required
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-400 mb-1">Notes (optional)</label>
                    <textarea
                      value={tradeForm.notes}
                      onChange={(e) => setTradeForm({ ...tradeForm, notes: e.target.value })}
                      className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-white"
                      rows={2}
                    />
                  </div>
                  <div className="flex gap-3 justify-end">
                    <button
                      type="button"
                      onClick={() => setShowCreateTrade(false)}
                      className="px-4 py-2 text-zinc-400 hover:text-white transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
                    >
                      Log Trade
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}

          {/* Trades List */}
          {loading ? (
            <div className="glass-card p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mx-auto"></div>
            </div>
          ) : trades.length === 0 ? (
            <div className="glass-card p-8 text-center">
              <p className="text-zinc-400">No trades logged yet. Start tracking your trades!</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-zinc-400 text-sm border-b border-zinc-700">
                    <th className="pb-3 font-medium">Asset</th>
                    <th className="pb-3 font-medium">Type</th>
                    <th className="pb-3 font-medium">Entry</th>
                    <th className="pb-3 font-medium">Exit</th>
                    <th className="pb-3 font-medium">P&L</th>
                    <th className="pb-3 font-medium">Status</th>
                    <th className="pb-3 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                  {trades.map((trade) => (
                    <tr key={trade.id} className="text-sm">
                      <td className="py-3">
                        <div>
                          <span className="text-white font-medium">{trade.asset.toUpperCase()}</span>
                          <span className="text-zinc-500 ml-1">({trade.market.toUpperCase()})</span>
                        </div>
                      </td>
                      <td className="py-3">
                        <span className={`px-2 py-1 rounded text-xs ${
                          trade.trade_type === 'long' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                          {trade.trade_type.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3 text-white font-mono">{formatCurrency(trade.entry_price)}</td>
                      <td className="py-3 text-white font-mono">
                        {trade.exit_price ? formatCurrency(trade.exit_price) : '-'}
                      </td>
                      <td className="py-3">
                        {trade.pnl !== null ? (
                          <span className={trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {formatCurrency(trade.pnl)}
                            {trade.pnl_percent !== null && (
                              <span className="text-xs ml-1">({formatPercent(trade.pnl_percent)})</span>
                            )}
                          </span>
                        ) : (
                          <span className="text-zinc-500">-</span>
                        )}
                      </td>
                      <td className="py-3">
                        <span className={`px-2 py-1 rounded text-xs ${
                          trade.status === 'open' ? 'bg-cyan-500/20 text-cyan-400' : 'bg-zinc-500/20 text-zinc-400'
                        }`}>
                          {trade.status}
                        </span>
                      </td>
                      <td className="py-3">
                        {trade.status === 'open' && (
                          closingTrade === trade.id ? (
                            <div className="flex items-center gap-2">
                              <input
                                type="number"
                                step="0.01"
                                value={exitPrice}
                                onChange={(e) => setExitPrice(e.target.value)}
                                placeholder="Exit price"
                                className="w-24 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-white text-xs"
                              />
                              <button
                                onClick={() => handleCloseTrade(trade.id)}
                                className="text-green-400 hover:text-green-300"
                              >
                                Close
                              </button>
                              <button
                                onClick={() => {
                                  setClosingTrade(null);
                                  setExitPrice('');
                                }}
                                className="text-zinc-400 hover:text-white"
                              >
                                Cancel
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={() => setClosingTrade(trade.id)}
                              className="text-cyan-400 hover:text-cyan-300 text-sm"
                            >
                              Close Trade
                            </button>
                          )
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
