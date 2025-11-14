'use client';

import { motion } from 'framer-motion';
import useSWR from 'swr';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { api, ROCCurve } from '@/lib/api';
import { BarChart3, TrendingUp, Download, RefreshCcw } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';

export default function Analytics() {
  // Fetch ROC curve data
  const { data: rocResponse, error, isLoading, mutate } = useSWR(
    '/api/analytics/roc',
    () => api.getROCCurves(),
    { refreshInterval: 0 }
  );

  const rocData = rocResponse?.curves;

  // Mock confusion matrix data (replace with real API call when available)
  const confusionMatrix = {
    tgn: { tp: 850, fp: 45, tn: 920, fn: 35 },
    mptgnn: { tp: 880, fp: 30, tn: 935, fn: 25 },
    baseline: { tp: 720, fp: 120, tn: 845, fn: 165 },
  };

  const getAccuracy = (cm: any) =>
    (((cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn)) * 100).toFixed(2);
  const getPrecision = (cm: any) =>
    ((cm.tp / (cm.tp + cm.fp)) * 100).toFixed(2);
  const getRecall = (cm: any) => ((cm.tp / (cm.tp + cm.fn)) * 100).toFixed(2);
  const getF1 = (cm: any) => {
    const precision = cm.tp / (cm.tp + cm.fp);
    const recall = cm.tp / (cm.tp + cm.fn);
    return (((2 * precision * recall) / (precision + recall)) * 100).toFixed(2);
  };

  // Model comparison data
  const modelComparison = [
    {
      model: 'MPTGNN',
      accuracy: parseFloat(getAccuracy(confusionMatrix.mptgnn)),
      precision: parseFloat(getPrecision(confusionMatrix.mptgnn)),
      recall: parseFloat(getRecall(confusionMatrix.mptgnn)),
      f1: parseFloat(getF1(confusionMatrix.mptgnn)),
      auc: rocData?.find((r) => r.model === 'mptgnn')?.auc || 0.94,
    },
    {
      model: 'TGN',
      accuracy: parseFloat(getAccuracy(confusionMatrix.tgn)),
      precision: parseFloat(getPrecision(confusionMatrix.tgn)),
      recall: parseFloat(getRecall(confusionMatrix.tgn)),
      f1: parseFloat(getF1(confusionMatrix.tgn)),
      auc: rocData?.find((r) => r.model === 'tgn')?.auc || 0.92,
    },
    {
      model: 'Baseline',
      accuracy: parseFloat(getAccuracy(confusionMatrix.baseline)),
      precision: parseFloat(getPrecision(confusionMatrix.baseline)),
      recall: parseFloat(getRecall(confusionMatrix.baseline)),
      f1: parseFloat(getF1(confusionMatrix.baseline)),
      auc: rocData?.find((r) => r.model === 'baseline')?.auc || 0.78,
    },
  ];

  const ConfusionMatrixCard = ({ title, data }: { title: string; data: any }) => (
    <div className="card">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="bg-green-500/20 p-4 rounded-lg text-center">
          <p className="text-2xl font-bold text-green-500">{data.tp}</p>
          <p className="text-xs text-muted-foreground">True Positive</p>
        </div>
        <div className="bg-red-500/20 p-4 rounded-lg text-center">
          <p className="text-2xl font-bold text-red-500">{data.fp}</p>
          <p className="text-xs text-muted-foreground">False Positive</p>
        </div>
        <div className="bg-red-500/20 p-4 rounded-lg text-center">
          <p className="text-2xl font-bold text-red-500">{data.fn}</p>
          <p className="text-xs text-muted-foreground">False Negative</p>
        </div>
        <div className="bg-green-500/20 p-4 rounded-lg text-center">
          <p className="text-2xl font-bold text-green-500">{data.tn}</p>
          <p className="text-xs text-muted-foreground">True Negative</p>
        </div>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Accuracy:</span>
          <span className="font-semibold">{getAccuracy(data)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Precision:</span>
          <span className="font-semibold">{getPrecision(data)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Recall:</span>
          <span className="font-semibold">{getRecall(data)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">F1 Score:</span>
          <span className="font-semibold">{getF1(data)}%</span>
        </div>
      </div>
    </div>
  );

  const actions = (
    <div className="flex gap-2">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => mutate()}
        className="px-4 py-2 bg-accent hover:bg-accent/80 rounded-lg flex items-center gap-2 transition-colors"
      >
        <RefreshCcw className="w-4 h-4" />
        Refresh
      </motion.button>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => alert('Export functionality coming soon!')}
        className="px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg flex items-center gap-2 transition-colors"
      >
        <Download className="w-4 h-4" />
        Export
      </motion.button>
    </div>
  );

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader
          title="Model Analytics"
          description="Performance metrics and comparison of fraud detection models"
          breadcrumbs={[{ label: 'Analytics' }]}
          actions={actions}
        />

      {/* ROC Curves */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <TrendingUp className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">ROC Curves</h2>
        </div>

        {isLoading && (
          <div className="h-96 flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-muted-foreground">Loading ROC data...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="h-96 flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <p className="text-red-500 mb-2">Failed to load analytics data</p>
              <p className="text-sm text-muted-foreground">
                Make sure the backend is running
              </p>
            </div>
          </div>
        )}

        {rocData && !isLoading && (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
                stroke="#888"
              />
              <YAxis
                dataKey="tpr"
                type="number"
                domain={[0, 1]}
                label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
                stroke="#888"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              {rocData.map((curve, index) => {
                // Create data points for the line
                const data = curve.fpr.map((fpr, i) => ({
                  fpr: fpr,
                  tpr: curve.tpr[i],
                }));
                const colors = ['#3b82f6', '#8b5cf6', '#ef4444'];
                return (
                  <Line
                    key={curve.model}
                    data={data}
                    type="monotone"
                    dataKey="tpr"
                    stroke={colors[index % colors.length]}
                    strokeWidth={2}
                    name={`${curve.model.toUpperCase()} (AUC: ${curve.auc.toFixed(3)})`}
                    dot={false}
                  />
                );
              })}
              {/* Diagonal reference line */}
              <Line
                data={[
                  { fpr: 0, tpr: 0 },
                  { fpr: 1, tpr: 1 },
                ]}
                type="monotone"
                dataKey="tpr"
                stroke="#666"
                strokeDasharray="5 5"
                strokeWidth={1}
                name="Random"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </motion.div>

      {/* Model Comparison Bar Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">Model Comparison</h2>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={modelComparison} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="model" stroke="#888" />
            <YAxis domain={[0, 100]} stroke="#888" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '8px',
              }}
            />
            <Legend />
            <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" />
            <Bar dataKey="precision" fill="#8b5cf6" name="Precision" />
            <Bar dataKey="recall" fill="#10b981" name="Recall" />
            <Bar dataKey="f1" fill="#f59e0b" name="F1 Score" />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Confusion Matrices */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold mb-4">Confusion Matrices</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <ConfusionMatrixCard title="MPTGNN" data={confusionMatrix.mptgnn} />
          <ConfusionMatrixCard title="TGN" data={confusionMatrix.tgn} />
          <ConfusionMatrixCard title="Baseline" data={confusionMatrix.baseline} />
        </div>
      </motion.div>

      {/* Performance Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <h2 className="text-2xl font-bold mb-4">Performance Summary</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4">Model</th>
                <th className="text-center py-3 px-4">Accuracy</th>
                <th className="text-center py-3 px-4">Precision</th>
                <th className="text-center py-3 px-4">Recall</th>
                <th className="text-center py-3 px-4">F1 Score</th>
                <th className="text-center py-3 px-4">AUC</th>
              </tr>
            </thead>
            <tbody>
              {modelComparison.map((model, index) => (
                <tr
                  key={model.model}
                  className="border-b border-border/50 hover:bg-accent/50 transition-colors"
                >
                  <td className="py-3 px-4 font-semibold">{model.model}</td>
                  <td className="text-center py-3 px-4">{model.accuracy.toFixed(2)}%</td>
                  <td className="text-center py-3 px-4">{model.precision.toFixed(2)}%</td>
                  <td className="text-center py-3 px-4">{model.recall.toFixed(2)}%</td>
                  <td className="text-center py-3 px-4">{model.f1.toFixed(2)}%</td>
                  <td className="text-center py-3 px-4">{(model.auc * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
      </div>
    </DashboardLayout>
  );
}
