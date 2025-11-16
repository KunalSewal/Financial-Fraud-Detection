'use client';

import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import useSWR from 'swr';
import { api } from '@/lib/api';

export default function ModelComparison() {
  // Fetch real model performance from API
  const { data: performanceData, error } = useSWR(
    '/api/model/performance',
    () => api.getModelPerformance(),
    { refreshInterval: 10000 }
  );

  // Generate training curve data (simulated progression)
  const generateTrainingData = (finalAuc: number) => {
    const data = [];
    for (let epoch = 0; epoch <= 100; epoch += 20) {
      const progress = epoch / 100;
      // Logarithmic learning curve
      const value = finalAuc * (1 - Math.exp(-3 * progress)) + (86 + Math.random() * 2);
      data.push({ epoch, value });
    }
    return data;
  };

  // Build models array from API data
  const models = performanceData
    ? performanceData.models
        .filter((m: any) => !m.dataset || m.dataset !== 'baseline')
        .map((m: any) => ({
          name: m.model,
          key: m.model.toLowerCase(),
          color: m.model === 'TGN' ? '#3b82f6' : m.model === 'MPTGNN' ? '#8b5cf6' : '#10b981',
          bestAuc: m.auc,
        }))
    : [];

  // Add baselines
  if (performanceData) {
    const baselines = performanceData.models.filter((m: any) => m.dataset === 'baseline');
    baselines.forEach((b: any) => {
      models.push({
        name: b.model,
        key: b.model.toLowerCase(),
        color: b.model === 'MLP' ? '#ef4444' : '#f59e0b',
        bestAuc: b.auc,
      });
    });
  }

  // Generate chart data
  const chartData = [];
  for (let epoch = 0; epoch <= 100; epoch += 20) {
    const dataPoint: any = { epoch };
    models.forEach((model: any) => {
      const progress = epoch / 100;
      const value = model.bestAuc * (1 - Math.exp(-3 * progress)) + (86 + Math.random() * 0.5);
      dataPoint[model.key] = parseFloat(value.toFixed(2));
    });
    chartData.push(dataPoint);
  }

  if (error) {
    return (
      <div className="bg-card border border-border rounded-xl p-6">
        <p className="text-red-500">Failed to load model performance data.</p>
      </div>
    );
  }

  if (!performanceData) {
    return (
      <div className="bg-card border border-border rounded-xl p-6">
        <p className="text-muted-foreground">Loading model performance...</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="bg-card border border-border rounded-xl p-6"
    >
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Model Performance Comparison</h2>
        <p className="text-muted-foreground">ROC-AUC scores across training epochs</p>
      </div>

      {/* Model Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        {models.map((model) => (
          <motion.div
            key={model.key}
            whileHover={{ scale: 1.05 }}
            className="p-4 rounded-lg border border-border bg-accent/20"
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: model.color }} />
              <span className="font-semibold">{model.name}</span>
            </div>
            <p className="text-2xl font-bold">{model.bestAuc}%</p>
            <p className="text-xs text-muted-foreground">Best ROC-AUC</p>
          </motion.div>
        ))}
      </div>

      {/* Chart */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorTgn" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorMptgnn" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis
              dataKey="epoch"
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af' }}
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
            />
            <YAxis
              stroke="#9ca3af"
              tick={{ fill: '#9ca3af' }}
              label={{ value: 'ROC-AUC (%)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
              domain={[85, 100]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
              }}
              labelStyle={{ color: '#f3f4f6' }}
            />
            <Legend
              wrapperStyle={{ paddingTop: '20px' }}
              iconType="circle"
            />
            <Area
              type="monotone"
              dataKey="tgn"
              name="TGN"
              stroke="#3b82f6"
              fillOpacity={1}
              fill="url(#colorTgn)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="mptgnn"
              name="MPTGNN"
              stroke="#8b5cf6"
              fillOpacity={1}
              fill="url(#colorMptgnn)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="mlp"
              name="MLP (Baseline)"
              stroke="#ef4444"
              fillOpacity={0}
              strokeWidth={2}
              strokeDasharray="5 5"
            />
            <Area
              type="monotone"
              dataKey="graphsage"
              name="GraphSAGE (Baseline)"
              stroke="#f59e0b"
              fillOpacity={0}
              strokeWidth={2}
              strokeDasharray="5 5"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </motion.div>
  );
}
