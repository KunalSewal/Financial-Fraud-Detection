'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import useSWR from 'swr';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { api, Dataset } from '@/lib/api';
import { Settings as SettingsIcon, Database, Save, RefreshCcw, CheckCircle } from 'lucide-react';

export default function Settings() {
  const [activeDataset, setActiveDataset] = useState('ibm');
  const [isSwitching, setIsSwitching] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');

  // Fetch available datasets
  const { data: datasetResponse, mutate } = useSWR(
    '/api/datasets',
    () => api.getDatasets(),
    { refreshInterval: 0 }
  );

  const datasets = datasetResponse?.datasets;

  const handleDatasetSwitch = async (datasetName: string) => {
    setIsSwitching(true);
    try {
      await api.switchDataset(datasetName);
      setActiveDataset(datasetName);
      setTimeout(() => setIsSwitching(false), 1000);
    } catch (error) {
      console.error('Failed to switch dataset:', error);
      setIsSwitching(false);
    }
  };

  const handleSaveSettings = () => {
    setSaveStatus('saving');
    setTimeout(() => {
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 2000);
    }, 1000);
  };

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader
          title="Settings"
          description="Configure system preferences and data sources"
          breadcrumbs={[{ label: 'Settings' }]}
        />

      {/* Dataset Selection */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Database className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">Dataset Configuration</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Active Dataset</label>
            <p className="text-sm text-muted-foreground mb-4">
              Select which dataset to use for fraud detection. Changes will affect all
              models and visualizations.
            </p>
          </div>

          {datasets ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {datasets.map((dataset) => (
                <motion.button
                  key={dataset.name}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => handleDatasetSwitch(dataset.name)}
                  disabled={isSwitching}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    activeDataset === dataset.name
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:border-primary/50'
                  } ${isSwitching ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-lg capitalize">
                        {dataset.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        {dataset.name === 'ibm'
                          ? 'IBM Credit Card Fraud Dataset (89,757 transactions, 33% fraud rate)'
                          : 'Unknown dataset'}
                      </p>
                    </div>
                    {activeDataset === dataset.name && (
                      <CheckCircle className="w-5 h-5 text-primary" />
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Nodes:</span>{' '}
                      <span className="font-semibold">
                        {dataset.num_nodes.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Edges:</span>{' '}
                      <span className="font-semibold">
                        {dataset.num_edges.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Features:</span>{' '}
                      <span className="font-semibold">{dataset.num_features}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Active:</span>{' '}
                      <span className={`font-semibold ${dataset.active ? 'text-green-500' : ''}`}>
                        {dataset.active ? 'Yes' : 'No'}
                      </span>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
              <p className="text-muted-foreground">Loading datasets...</p>
            </div>
          )}

          {isSwitching && (
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <RefreshCcw className="w-5 h-5 text-blue-500 animate-spin" />
                <p className="text-sm">
                  Switching dataset... This may take a few moments.
                </p>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Model Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <SettingsIcon className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">Model Configuration</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">
              Fraud Detection Threshold
            </label>
            <input
              type="range"
              min="0"
              max="100"
              defaultValue="70"
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Low (0%)</span>
              <span>Medium (50%)</span>
              <span>High (100%)</span>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">
              Model Update Frequency
            </label>
            <select className="w-full bg-accent rounded-lg px-4 py-2 border border-border">
              <option>Real-time (Live)</option>
              <option>Every 5 minutes</option>
              <option>Every 15 minutes</option>
              <option>Hourly</option>
              <option>Daily</option>
            </select>
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">
              Auto-retrain on New Data
            </label>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="auto-retrain"
                defaultChecked
                className="w-4 h-4"
              />
              <label htmlFor="auto-retrain" className="text-sm text-muted-foreground">
                Automatically retrain models when new data is available
              </label>
            </div>
          </div>
        </div>
      </motion.div>

      {/* API Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Database className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">API Configuration</h2>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Backend URL</label>
            <input
              type="text"
              defaultValue="http://localhost:8000"
              className="w-full bg-accent rounded-lg px-4 py-2 border border-border"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">API Key</label>
            <input
              type="password"
              placeholder="Enter API key (optional)"
              className="w-full bg-accent rounded-lg px-4 py-2 border border-border"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Request Timeout</label>
            <select className="w-full bg-accent rounded-lg px-4 py-2 border border-border">
              <option>5 seconds</option>
              <option>10 seconds</option>
              <option>30 seconds</option>
              <option>60 seconds</option>
            </select>
          </div>
        </div>
      </motion.div>

      {/* Save Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex justify-end gap-3"
      >
        <button
          onClick={() => window.location.reload()}
          className="px-6 py-3 bg-accent hover:bg-accent/80 rounded-lg flex items-center gap-2 transition-colors"
        >
          <RefreshCcw className="w-4 h-4" />
          Reset
        </button>
        <button
          onClick={handleSaveSettings}
          disabled={saveStatus === 'saving'}
          className={`px-6 py-3 rounded-lg flex items-center gap-2 transition-colors ${
            saveStatus === 'saved'
              ? 'bg-green-500 hover:bg-green-600'
              : 'bg-primary hover:bg-primary/90'
          } text-primary-foreground`}
        >
          {saveStatus === 'saving' ? (
            <>
              <RefreshCcw className="w-4 h-4 animate-spin" />
              Saving...
            </>
          ) : saveStatus === 'saved' ? (
            <>
              <CheckCircle className="w-4 h-4" />
              Saved!
            </>
          ) : (
            <>
              <Save className="w-4 h-4" />
              Save Settings
            </>
          )}
        </button>
      </motion.div>
      </div>
    </DashboardLayout>
  );
}
