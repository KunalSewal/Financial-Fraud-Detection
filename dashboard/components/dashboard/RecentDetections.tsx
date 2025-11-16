'use client';

import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import useSWR from 'swr';
import { api } from '@/lib/api';

export default function RecentDetections() {
  // Fetch real transactions from API
  const { data: transactionsData, error } = useSWR(
    '/api/transactions/recent',
    () => api.getRecentTransactions(5),
    { refreshInterval: 5000 } // Update every 5 seconds
  );

  const detections = transactionsData?.transactions || [];

  const riskColors = {
    high: { bg: 'bg-red-500/10', text: 'text-red-500', border: 'border-red-500/30' },
    medium: { bg: 'bg-yellow-500/10', text: 'text-yellow-500', border: 'border-yellow-500/30' },
    low: { bg: 'bg-green-500/10', text: 'text-green-500', border: 'border-green-500/30' },
  };

  const statusIcons = {
    blocked: <AlertTriangle className="w-4 h-4" />,
    flagged: <Clock className="w-4 h-4" />,
    reviewing: <CheckCircle className="w-4 h-4" />,
    approved: <CheckCircle className="w-4 h-4" />,
  };

  if (error) {
    return (
      <div className="bg-card border border-border rounded-xl p-6">
        <p className="text-red-500">Failed to load transactions.</p>
      </div>
    );
  }

  if (!transactionsData) {
    return (
      <div className="bg-card border border-border rounded-xl p-6">
        <p className="text-muted-foreground">Loading transactions...</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
      className="bg-card border border-border rounded-xl p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold mb-2">Recent Fraud Detections</h2>
          <p className="text-muted-foreground">Latest suspicious transactions flagged by TGN</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
        >
          View All
        </motion.button>
      </div>

      <div className="space-y-3">
        {detections.map((detection, index) => {
          const riskStyle = riskColors[detection.risk];

          return (
            <motion.div
              key={detection.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 + index * 0.1 }}
              whileHover={{ x: 5, backgroundColor: 'rgba(59, 130, 246, 0.05)' }}
              className="flex items-center gap-4 p-4 rounded-lg border border-border hover:border-primary/50 transition-all cursor-pointer"
            >
              {/* Transaction ID & Time */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono font-semibold">{detection.id}</span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(detection.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground truncate">
                  {detection.source} → {detection.target}
                </p>
              </div>

              {/* Amount */}
              <div className="text-right">
                <p className="font-bold text-lg">${detection.amount.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">
                  {(detection.fraud_probability * 100).toFixed(1)}% fraud prob
                </p>
              </div>

              {/* Risk Level */}
              <div
                className={`px-3 py-1.5 rounded-full ${riskStyle.bg} border ${riskStyle.border}`}
              >
                <span className={`text-xs font-semibold uppercase ${riskStyle.text}`}>
                  {detection.risk} risk
                </span>
              </div>

              {/* Status */}
              <div className="flex items-center gap-2">
                <div className={riskStyle.text}>{statusIcons[detection.status]}</div>
                <span className="text-sm font-medium capitalize">{detection.status}</span>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Footer Stats - Calculate from actual transaction data */}
      {transactionsData && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-6 pt-6 border-t border-border grid grid-cols-3 gap-4"
        >
          <div className="text-center">
            <p className="text-2xl font-bold text-red-500">
              {detections.filter(d => d.status === 'blocked').length}
            </p>
            <p className="text-xs text-muted-foreground">Blocked (Showing)</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-yellow-500">
              {detections.filter(d => d.status === 'reviewing' || d.status === 'flagged').length}
            </p>
            <p className="text-xs text-muted-foreground">Under Review</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-500">
              {detections.filter(d => d.risk === 'high' || d.risk === 'medium').length}
            </p>
            <p className="text-xs text-muted-foreground">High Risk Detected</p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
