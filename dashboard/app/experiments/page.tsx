'use client';

import { motion } from 'framer-motion';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { TrendingUp, Clock, Zap, Database, Download, ExternalLink } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function Experiments() {
  // Mock training history data
  const trainingHistory = {
    tgn: Array.from({ length: 50 }, (_, i) => ({
      epoch: i + 1,
      train_loss: 0.5 * Math.exp(-i / 15) + 0.05,
      val_loss: 0.52 * Math.exp(-i / 15) + 0.08,
      train_auc: 0.5 + 0.42 * (1 - Math.exp(-i / 12)),
      val_auc: 0.5 + 0.4 * (1 - Math.exp(-i / 12)),
    })),
    mptgnn: Array.from({ length: 50 }, (_, i) => ({
      epoch: i + 1,
      train_loss: 0.48 * Math.exp(-i / 14) + 0.04,
      val_loss: 0.5 * Math.exp(-i / 14) + 0.07,
      train_auc: 0.5 + 0.44 * (1 - Math.exp(-i / 11)),
      val_auc: 0.5 + 0.42 * (1 - Math.exp(-i / 11)),
    })),
  };

  const experiments = [
    {
      id: 1,
      name: 'TGN - Ethereum Dataset',
      status: 'completed',
      duration: '2h 34m',
      bestAUC: 0.92,
      date: '2025-11-08',
    },
    {
      id: 2,
      name: 'MPTGNN - Ethereum Dataset',
      status: 'completed',
      duration: '3h 12m',
      bestAUC: 0.94,
      date: '2025-11-08',
    },
    {
      id: 3,
      name: 'TGN - DGraph Dataset',
      status: 'running',
      duration: '1h 45m',
      bestAUC: 0.89,
      date: '2025-11-10',
    },
  ];

  const actions = (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => window.open('https://wandb.ai', '_blank')}
      className="px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg flex items-center gap-2 transition-colors"
    >
      <ExternalLink className="w-4 h-4" />
      Open W&B
    </motion.button>
  );

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader
          title="Experiments"
          description="Training history and experiment tracking"
          breadcrumbs={[{ label: 'Experiments' }]}
          actions={actions}
        />

      {/* Experiment Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {experiments.map((exp, index) => (
          <motion.div
            key={exp.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <h3 className="font-semibold mb-1">{exp.name}</h3>
                <p className="text-xs text-muted-foreground">{exp.date}</p>
              </div>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${
                  exp.status === 'completed'
                    ? 'bg-green-500/20 text-green-500'
                    : exp.status === 'running'
                    ? 'bg-blue-500/20 text-blue-500 animate-pulse'
                    : 'bg-red-500/20 text-red-500'
                }`}
              >
                {exp.status}
              </span>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Duration
                </span>
                <span className="font-medium">{exp.duration}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Best AUC
                </span>
                <span className="font-medium">{(exp.bestAUC * 100).toFixed(2)}%</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* TGN Training Curves */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <TrendingUp className="w-6 h-6 text-blue-500" />
          <h2 className="text-2xl font-bold">TGN Training History</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Loss Curve */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
              Loss over Epochs
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingHistory.tgn}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="epoch" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="Train Loss"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  name="Val Loss"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* AUC Curve */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
              AUC over Epochs
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingHistory.tgn}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="epoch" stroke="#888" />
                <YAxis domain={[0.5, 1]} stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_auc"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Train AUC"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_auc"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  name="Val AUC"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* MPTGNN Training Curves */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <TrendingUp className="w-6 h-6 text-purple-500" />
          <h2 className="text-2xl font-bold">MPTGNN Training History</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Loss Curve */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
              Loss over Epochs
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingHistory.mptgnn}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="epoch" stroke="#888" />
                <YAxis stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="Train Loss"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  name="Val Loss"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* AUC Curve */}
          <div>
            <h3 className="text-sm font-semibold mb-3 text-muted-foreground">
              AUC over Epochs
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingHistory.mptgnn}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="epoch" stroke="#888" />
                <YAxis domain={[0.5, 1]} stroke="#888" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_auc"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="Train AUC"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="val_auc"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  name="Val AUC"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      {/* W&B Integration Info */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="card bg-blue-500/10 border-blue-500/20"
      >
        <div className="flex items-start gap-3">
          <Database className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold mb-1">Weights & Biases Integration</h4>
            <p className="text-sm text-muted-foreground mb-3">
              Track your experiments with W&B for detailed metrics, artifacts, and
              collaboration. The training data shown above is simulated - connect to W&B
              for real-time experiment tracking.
            </p>
            <button
              onClick={() => window.open('https://wandb.ai', '_blank')}
              className="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm flex items-center gap-2 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              Visit W&B Dashboard
            </button>
          </div>
        </div>
      </motion.div>
      </div>
    </DashboardLayout>
  );
}
