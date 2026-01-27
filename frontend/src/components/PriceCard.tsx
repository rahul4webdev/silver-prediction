'use client';

import { useEffect, useState, useRef } from 'react';
import { getLivePrice, getMCXSilverContracts } from '@/lib/api';
import { formatCurrency, formatPercent, cn } from '@/lib/utils';
import { usePriceUpdates } from '@/hooks/useWebSocket';
import type { LivePrice, Asset, ContractInfo, ContractType } from '@/lib/types';
import LatestPredictions from './LatestPredictions';

interface PriceCardProps {
  asset?: Asset;
  market: 'mcx' | 'comex';
}

// Contract type labels for display
const CONTRACT_LABELS: Record<ContractType, string> = {
  'SILVER': 'Silver (30kg)',
  'SILVERM': 'Silver Mini (5kg)',
  'SILVERMIC': 'Silver Micro (1kg)',
};

export default function PriceCard({ asset = 'silver', market }: PriceCardProps) {
  const [price, setPrice] = useState<LivePrice | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Contract selection state (MCX only)
  const [contracts, setContracts] = useState<ContractInfo[]>([]);
  const [selectedContract, setSelectedContract] = useState<ContractInfo | null>(null);
  const [contractsLoading, setContractsLoading] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Use WebSocket for MCX real-time updates (now receives all contract prices)
  const { price: wsPrice, prices: wsPrices, getPriceBySymbol, isConnected, connectionStatus, send } = usePriceUpdates(asset, market);

  // Fetch MCX contracts on mount
  useEffect(() => {
    if (market !== 'mcx' || asset !== 'silver') return;

    async function fetchContracts() {
      setContractsLoading(true);
      try {
        const response = await getMCXSilverContracts();
        if (response && response.status === 'success') {
          setContracts(response.contracts);
          // Find default contract (first SILVER contract or first available)
          const defaultContract = response.contracts.find(c => c.contract_type === 'SILVER')
            || response.contracts[0];
          if (defaultContract && !selectedContract) {
            setSelectedContract(defaultContract);
          }
        }
      } catch (err) {
        console.error('Failed to fetch contracts:', err);
      } finally {
        setContractsLoading(false);
      }
    }

    fetchContracts();
  }, [market, asset]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch initial price via REST API
  useEffect(() => {
    async function fetchInitialPrice() {
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

    fetchInitialPrice();
  }, [asset, market]);

  // Update price from WebSocket for MCX - filter by selected contract
  useEffect(() => {
    if (market !== 'mcx') return;

    // Get price for the selected contract
    const contractPrice = selectedContract
      ? getPriceBySymbol(selectedContract.instrument_key)
      : wsPrice;

    if (contractPrice && contractPrice.market === 'mcx') {
      setPrice(prev => ({
        ...prev,
        asset: contractPrice.asset,
        market: contractPrice.market,
        symbol: contractPrice.symbol,
        price: contractPrice.price,
        open: contractPrice.open ?? prev?.open,
        high: contractPrice.high ?? prev?.high,
        low: contractPrice.low ?? prev?.low,
        change: contractPrice.change ?? prev?.change,
        change_percent: contractPrice.change_percent ?? prev?.change_percent,
        volume: contractPrice.volume ?? prev?.volume,
        timestamp: contractPrice.timestamp,
        source: 'upstox_ws',
      }));
      setError(null);
      setLoading(false);
    }
  }, [wsPrice, wsPrices, selectedContract, getPriceBySymbol, market]);

  // Request all contract prices when connected and when contract changes
  useEffect(() => {
    if (market === 'mcx' && isConnected && selectedContract) {
      // Request price for the selected contract if we don't have it yet
      const existingPrice = getPriceBySymbol(selectedContract.instrument_key);
      if (!existingPrice) {
        // Request all contract prices from the WebSocket
        send({ action: 'get_all_contracts', market: 'mcx' });
      }
    }
  }, [market, isConnected, selectedContract, getPriceBySymbol, send]);

  // Fallback polling for COMEX (since we don't have real-time WebSocket for it)
  useEffect(() => {
    if (market !== 'comex') return;

    const interval = setInterval(async () => {
      try {
        const data = await getLivePrice(asset, market);
        if (data.status !== 'error') {
          setPrice(data);
          setError(null);
        }
      } catch (err) {
        // Keep existing price on error
      }
    }, 5000); // Poll every 5 seconds for COMEX

    return () => clearInterval(interval);
  }, [asset, market]);

  const isPositive = (price?.change_percent ?? 0) >= 0;
  const currency = market === 'mcx' ? 'INR' : 'USD';

  if (loading && !price) {
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

  // Group contracts by type for dropdown
  const groupedContracts = contracts.reduce((acc, contract) => {
    if (!acc[contract.contract_type]) {
      acc[contract.contract_type] = [];
    }
    acc[contract.contract_type].push(contract);
    return acc;
  }, {} as Record<ContractType, ContractInfo[]>);

  // Get display label for selected contract
  const getContractDisplayLabel = (contract: ContractInfo | null) => {
    if (!contract) return 'Select Contract';
    return `${contract.contract_type} ${contract.expiry_date || ''}`.trim();
  };

  if (error && !price) {
    return (
      <div className="glass-card p-4 sm:p-6">
        <div className="text-zinc-400 text-sm">
          {market.toUpperCase()} {asset.charAt(0).toUpperCase() + asset.slice(1)}
          {market === 'mcx' && asset === 'silver' && selectedContract && (
            <span className="text-zinc-500"> ({selectedContract.contract_type} {selectedContract.expiry_date})</span>
          )}
        </div>
        <div className="text-zinc-500 mt-2">Unable to load price</div>
        <LatestPredictions asset={asset} market={market} />
      </div>
    );
  }

  if (!price) return null;

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
          <div className="flex items-center gap-2">
            <span className="text-zinc-400 text-xs sm:text-sm font-medium">
              {market.toUpperCase()} {asset.charAt(0).toUpperCase() + asset.slice(1)}
            </span>
            {/* Contract dropdown for MCX Silver */}
            {market === 'mcx' && asset === 'silver' && contracts.length > 0 && (
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  className="flex items-center gap-1 text-[10px] sm:text-xs text-zinc-400 bg-white/5 hover:bg-white/10 px-2 py-1 rounded transition-colors"
                >
                  <span className="truncate max-w-[100px] sm:max-w-[140px]">
                    {getContractDisplayLabel(selectedContract)}
                  </span>
                  <svg
                    className={cn("w-3 h-3 transition-transform", dropdownOpen && "rotate-180")}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {dropdownOpen && (
                  <div className="absolute left-0 top-full mt-1 w-56 bg-zinc-900 border border-white/10 rounded-lg shadow-xl z-50 max-h-64 overflow-y-auto">
                    {(['SILVER', 'SILVERM', 'SILVERMIC'] as ContractType[]).map(type => (
                      groupedContracts[type] && groupedContracts[type].length > 0 && (
                        <div key={type}>
                          <div className="px-3 py-2 text-[10px] text-zinc-500 font-medium bg-white/5 sticky top-0">
                            {CONTRACT_LABELS[type]}
                          </div>
                          {groupedContracts[type].map(contract => (
                            <button
                              key={contract.instrument_key}
                              onClick={() => {
                                setSelectedContract(contract);
                                setDropdownOpen(false);
                              }}
                              className={cn(
                                "w-full text-left px-3 py-2 text-xs hover:bg-white/10 transition-colors flex items-center justify-between",
                                selectedContract?.instrument_key === contract.instrument_key && "bg-white/5 text-white"
                              )}
                            >
                              <span>
                                {contract.contract_type} {contract.expiry_date || 'No Expiry'}
                              </span>
                              {selectedContract?.instrument_key === contract.instrument_key && (
                                <svg className="w-3 h-3 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                </svg>
                              )}
                            </button>
                          ))}
                        </div>
                      )
                    ))}
                  </div>
                )}
              </div>
            )}
            {/* Show loading indicator while fetching contracts */}
            {market === 'mcx' && asset === 'silver' && contractsLoading && (
              <span className="text-[10px] text-zinc-500">Loading contracts...</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* WebSocket connection indicator for MCX */}
            {market === 'mcx' && (
              <span className={cn(
                "w-2 h-2 rounded-full",
                isConnected ? "bg-green-500" : "bg-yellow-500 animate-pulse"
              )} title={isConnected ? "Live" : "Connecting..."} />
            )}
            <span className="text-[10px] sm:text-xs text-zinc-500 bg-white/5 px-2 py-1 rounded truncate max-w-[80px] sm:max-w-none">
              {price.source?.replace('_', ' ')}
            </span>
          </div>
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
            {selectedContract && selectedContract.lot_size && (
              <span className="ml-2 text-zinc-600">
                (Lot: {selectedContract.lot_size} kg)
              </span>
            )}
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
