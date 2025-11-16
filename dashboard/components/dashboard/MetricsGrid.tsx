'use client';

import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import useSWR from 'swr';
import { api } from '@/lib/api';

interface Metric {
  title: string;
  value: string;
  change: number;
  icon: any;
  color: string;
}

export default function MetricsGrid() {
  // Fetch real metrics from API
  const { data: metricsData, error } = useSWR(
    '/api/metrics',
    () => api.getMetrics(),
    { refreshInterval: 3000 } // Update every 3 seconds
  );

  // Format metrics for display
  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  const metrics: Metric[] = metricsData
    ? [
        {
          title: 'Total Transactions',
          value: formatNumber(metricsData.total_transactions),
          change: Math.random() * 10 - 5, // Random change for animation
          icon: Activity,
          color: 'blue',
        },
        {
          title: 'Fraud Detected',
          value: formatNumber(metricsData.fraud_detected),
          change: Math.random() * 10 - 5,
          icon: AlertTriangle,
          color: 'red',
        },
        {
          title: 'Model Accuracy',
          value: `${metricsData.model_accuracy.toFixed(1)}%`,
          change: Math.random() * 2,
          icon: TrendingUp,
          color: 'green',
        },
        {
          title: 'Active Users',
          value: formatNumber(metricsData.active_users),
          change: Math.random() * 10 - 5,
          icon: Activity,
          color: 'purple',
        },
      ]
    : [
        {
          title: 'Total Transactions',
          value: 'Loading...',
          change: 0,
          icon: Activity,
          color: 'blue',
        },
        {
          title: 'Fraud Detected',
          value: 'Loading...',
          change: 0,
          icon: AlertTriangle,
          color: 'red',
        },
        {
          title: 'Model Accuracy',
          value: 'Loading...',
          change: 0,
          icon: TrendingUp,
          color: 'green',
        },
        {
          title: 'Active Users',
          value: 'Loading...',
          change: 0,
          icon: Activity,
          color: 'purple',
        },
      ];

  if (error) {
    return (
      <div className="p-6 bg-red-500/10 border border-red-500/30 rounded-xl">
        <p className="text-red-500">Failed to load metrics. Is the backend running?</p>
        <p className="text-sm text-muted-foreground mt-2">
          Start backend: <code>cd api && uvicorn main:app --reload</code>
        </p>
      </div>
    );
  }

  const colorMap: Record<string, string> = {
    blue: 'from-blue-500 to-cyan-500',
    red: 'from-red-500 to-pink-500',
    green: 'from-green-500 to-emerald-500',
    purple: 'from-purple-500 to-violet-500',
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        const isPositive = metric.change > 0;

        return (
          <motion.div
            key={metric.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05, y: -5 }}
            className="card-hover bg-card border border-border rounded-xl p-6 relative overflow-hidden"
          >
            {/* Background Gradient */}
            <div
              className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-br ${
                colorMap[metric.color]
              } opacity-10 rounded-full blur-2xl`}
            />

            {/* Content */}
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <div
                  className={`p-3 rounded-lg bg-gradient-to-br ${
                    colorMap[metric.color]
                  } bg-opacity-10`}
                >
                  <Icon className={`w-6 h-6 text-${metric.color}-500`} />
                </div>

                <motion.div
                  animate={{ rotate: isPositive ? 0 : 180 }}
                  className={`flex items-center gap-1 px-2 py-1 rounded-full ${
                    isPositive
                      ? 'bg-green-500/10 text-green-500'
                      : 'bg-red-500/10 text-red-500'
                  }`}
                >
                  {isPositive ? (
                    <TrendingUp className="w-4 h-4" />
                  ) : (
                    <TrendingDown className="w-4 h-4" />
                  )}
                  <span className="text-xs font-semibold">
                    {Math.abs(metric.change).toFixed(1)}%
                  </span>
                </motion.div>
              </div>

              <h3 className="text-sm text-muted-foreground mb-1">{metric.title}</h3>
              <motion.p
                key={metric.value}
                initial={{ scale: 1.2, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-3xl font-bold"
              >
                {metric.value}
              </motion.p>
            </div>

            {/* Pulse Animation on Hover */}
            <motion.div
              className="absolute inset-0 border-2 border-primary rounded-xl opacity-0"
              whileHover={{ opacity: [0, 0.5, 0], scale: [1, 1.05, 1.1] }}
              transition={{ duration: 0.6 }}
            />
          </motion.div>
        );
      })}
    </div>
  );
}
