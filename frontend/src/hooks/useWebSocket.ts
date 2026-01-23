'use client';

import { useEffect, useState, useRef, useCallback } from 'react';

interface WebSocketMessage {
  type: string;
  [key: string]: unknown;
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastMessage: WebSocketMessage | null;
  send: (message: object) => void;
  reconnect: () => void;
}

// Dedicated WebSocket server on port 8024 (bypasses OpenLiteSpeed proxy limitations)
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'wss://predictionapi.gahfaudio.in:8024';

export function useWebSocket(
  endpoint: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnectInterval = 5000,
    maxReconnectAttempts = 5,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    try {
      // Clean up existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      const url = `${WS_URL}${endpoint}`;
      console.log('[WebSocket] Connecting to:', url);

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        setIsConnected(true);
        reconnectAttemptsRef.current = 0;
        onConnect?.();

        // Send subscription message
        ws.send(JSON.stringify({ action: 'subscribe' }));
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          setLastMessage(message);
          onMessage?.(message);
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        onError?.(error);
      };

      ws.onclose = () => {
        console.log('[WebSocket] Disconnected');
        setIsConnected(false);
        onDisconnect?.();

        // Attempt to reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          console.log(
            `[WebSocket] Reconnecting in ${reconnectInterval}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`
          );
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
        } else {
          console.log('[WebSocket] Max reconnect attempts reached');
        }
      };
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);
    }
  }, [endpoint, onConnect, onDisconnect, onError, onMessage, reconnectInterval, maxReconnectAttempts]);

  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Cannot send message - not connected');
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0;
    connect();
  }, [connect]);

  useEffect(() => {
    connect();

    // Set up ping interval to keep connection alive
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ action: 'ping' }));
      }
    }, 30000);

    return () => {
      clearInterval(pingInterval);
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    isConnected,
    lastMessage,
    send,
    reconnect,
  };
}

// Hook for price updates
export interface PriceData {
  asset: string;
  market: string;
  symbol: string;
  price: number;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  change: number | null;
  change_percent: number | null;
  volume: number | null;
  timestamp: string;
  source: string;
}

export function usePriceUpdates(asset: string = 'silver', market: string = 'mcx') {
  const [price, setPrice] = useState<PriceData | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');

  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'price_update') {
      // Only update if it matches our market filter
      if (message.market === market || market === 'all') {
        setPrice({
          asset: message.asset as string,
          market: message.market as string,
          symbol: message.symbol as string,
          price: message.price as number,
          open: (message.open as number) ?? null,
          high: (message.high as number) ?? null,
          low: (message.low as number) ?? null,
          close: (message.close as number) ?? null,
          change: (message.change as number) ?? null,
          change_percent: (message.change_percent as number) ?? null,
          volume: (message.volume as number) ?? null,
          timestamp: message.timestamp as string,
          source: (message.source as string) ?? 'websocket',
        });
      }
    } else if (message.type === 'connected') {
      setConnectionStatus('connected');
    }
  }, [market]);

  const handleConnect = useCallback(() => {
    setConnectionStatus('connected');
  }, []);

  const handleDisconnect = useCallback(() => {
    setConnectionStatus('disconnected');
  }, []);

  const { isConnected, lastMessage, send, reconnect } = useWebSocket(
    `/ws/prices/${asset}`,
    {
      onMessage: handleMessage,
      onConnect: handleConnect,
      onDisconnect: handleDisconnect,
    }
  );

  return {
    price,
    isConnected,
    connectionStatus,
    lastMessage,
    send,
    reconnect,
  };
}

// Hook for prediction updates
export function usePredictionUpdates() {
  const [latestPrediction, setLatestPrediction] = useState<{
    id: string;
    asset: string;
    market: string;
    interval: string;
    direction: string;
    confidence: number;
    predictedPrice: number;
    currentPrice: number;
    targetTime: string;
  } | null>(null);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    if (message.type === 'new_prediction') {
      setLatestPrediction({
        id: message.id as string,
        asset: message.asset as string,
        market: message.market as string,
        interval: message.interval as string,
        direction: message.direction as string,
        confidence: message.confidence as number,
        predictedPrice: message.predicted_price as number,
        currentPrice: message.current_price as number,
        targetTime: message.target_time as string,
      });
    }
  }, []);

  const { isConnected, lastMessage, send, reconnect } = useWebSocket(
    '/ws/predictions',
    { onMessage: handleMessage }
  );

  return {
    latestPrediction,
    isConnected,
    lastMessage,
    send,
    reconnect,
  };
}
