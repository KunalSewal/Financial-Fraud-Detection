'use client';

import { motion } from 'framer-motion';
import useSWR from 'swr';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { api, Transaction } from '@/lib/api';
import { Bell, AlertTriangle, CheckCircle, XCircle, Clock } from 'lucide-react';
import { useState } from 'react';

export default function Alerts() {
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');

  // Fetch recent high-risk transactions as alerts
  const { data: transactionData, isLoading } = useSWR(
    '/api/transactions/recent',
    () => api.getRecentTransactions(50),
    { refreshInterval: 5000 }
  );

  const transactions = transactionData?.transactions;

  const alerts = transactions
    ?.filter((t) => t.fraud_probability > 0.3) // Only show potential fraud
    .map((t) => ({
      id: t.id,
      timestamp: new Date().toISOString(),
      type: 'fraud_detection',
      severity: t.risk,
      message: `Suspicious transaction detected from ${t.source} to ${t.target}`,
      amount: t.amount,
      probability: t.fraud_probability,
      status: t.status,
    }));

  const filteredAlerts =
    filter === 'all'
      ? alerts
      : alerts?.filter((a) => a.severity === filter);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'medium':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      default:
        return <CheckCircle className="w-5 h-5 text-green-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'bg-red-500/10 border-red-500/20';
      case 'medium':
        return 'bg-yellow-500/10 border-yellow-500/20';
      default:
        return 'bg-green-500/10 border-green-500/20';
    }
  };

  const actions = (
    <div className="flex gap-2">
      <button
        onClick={() => setFilter('all')}
        className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
          filter === 'all'
            ? 'bg-primary text-primary-foreground'
            : 'bg-accent hover:bg-accent/80'
        }`}
      >
        All
      </button>
      <button
        onClick={() => setFilter('high')}
        className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
          filter === 'high'
            ? 'bg-red-500 text-white'
            : 'bg-accent hover:bg-accent/80'
        }`}
      >
        High
      </button>
      <button
        onClick={() => setFilter('medium')}
        className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
          filter === 'medium'
            ? 'bg-yellow-500 text-white'
            : 'bg-accent hover:bg-accent/80'
        }`}
      >
        Medium
      </button>
      <button
        onClick={() => setFilter('low')}
        className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
          filter === 'low'
            ? 'bg-green-500 text-white'
            : 'bg-accent hover:bg-accent/80'
        }`}
      >
        Low
      </button>
    </div>
  );

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader
          title="Alerts & Notifications"
          description="Real-time fraud detection alerts and system notifications"
          breadcrumbs={[{ label: 'Alerts' }]}
          actions={actions}
        />

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Alerts</p>
              <p className="text-2xl font-bold">{alerts?.length || 0}</p>
            </div>
            <Bell className="w-8 h-8 text-blue-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">High Severity</p>
              <p className="text-2xl font-bold text-red-500">
                {alerts?.filter((a) => a.severity === 'high').length || 0}
              </p>
            </div>
            <XCircle className="w-8 h-8 text-red-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Medium Severity</p>
              <p className="text-2xl font-bold text-yellow-500">
                {alerts?.filter((a) => a.severity === 'medium').length || 0}
              </p>
            </div>
            <AlertTriangle className="w-8 h-8 text-yellow-500" />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Low Severity</p>
              <p className="text-2xl font-bold text-green-500">
                {alerts?.filter((a) => a.severity === 'low').length || 0}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </motion.div>
      </div>

      {/* Alerts List */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Bell className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">Recent Alerts</h2>
        </div>

        {isLoading && (
          <div className="text-center py-12">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-muted-foreground">Loading alerts...</p>
          </div>
        )}

        {!isLoading && filteredAlerts && filteredAlerts.length === 0 && (
          <div className="text-center py-12">
            <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <p className="text-muted-foreground">No alerts to display</p>
          </div>
        )}

        <div className="space-y-3">
          {filteredAlerts?.map((alert, index) => (
            <motion.div
              key={alert.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)}`}
            >
              <div className="flex items-start gap-4">
                {getSeverityIcon(alert.severity)}
                <div className="flex-1">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-semibold">{alert.message}</h3>
                      <p className="text-sm text-muted-foreground flex items-center gap-2 mt-1">
                        <Clock className="w-3 h-3" />
                        {new Date(alert.timestamp).toLocaleString()}
                      </p>
                    </div>
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium uppercase ${
                        alert.severity === 'high'
                          ? 'bg-red-500/20 text-red-500'
                          : alert.severity === 'medium'
                          ? 'bg-yellow-500/20 text-yellow-500'
                          : 'bg-green-500/20 text-green-500'
                      }`}
                    >
                      {alert.severity}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <div>
                      <span className="text-muted-foreground">Amount:</span>{' '}
                      <span className="font-semibold">
                        ${alert.amount.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Probability:</span>{' '}
                      <span className="font-semibold">
                        {(alert.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Status:</span>{' '}
                      <span
                        className={`font-semibold ${
                          alert.status === 'blocked' || alert.status === 'flagged'
                            ? 'text-red-500'
                            : 'text-green-500'
                        }`}
                      >
                        {alert.status}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
      </div>
    </DashboardLayout>
  );
}
