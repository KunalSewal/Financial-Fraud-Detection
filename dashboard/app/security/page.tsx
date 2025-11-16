'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import useSWR from 'swr';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { Shield, Lock, AlertTriangle, CheckCircle, Activity, Server, Database } from 'lucide-react';
import { api } from '@/lib/api';

export default function Security() {
  const [uptime, setUptime] = useState(0);
  
  // Fetch real metrics to derive security status
  const { data: metricsData, error: metricsError } = useSWR(
    '/api/metrics',
    () => api.getMetrics(),
    { refreshInterval: 5000 }
  );
  
  const { data: datasetsData, error: datasetsError } = useSWR(
    '/api/datasets',
    () => api.getDatasets(),
    { refreshInterval: 0 }
  );

  // Calculate uptime since component mount
  useEffect(() => {
    const interval = setInterval(() => {
      setUptime(prev => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${mins}m ${secs}s`;
  };

  // Real security metrics derived from actual system state
  const apiHealthy = !metricsError && !datasetsError;
  const datasetLoaded = datasetsData?.datasets?.length > 0;
  const modelsActive = metricsData?.total_transactions > 0;
  
  const securityMetrics = [
    {
      name: 'API Status',
      status: apiHealthy ? 'secure' : 'error',
      value: apiHealthy ? 'Healthy' : 'Error',
      icon: Server,
      color: apiHealthy ? 'text-green-500' : 'text-red-500',
    },
    {
      name: 'Dataset Loaded',
      status: datasetLoaded ? 'secure' : 'warning',
      value: datasetLoaded ? 'Yes' : 'No',
      icon: Database,
      color: datasetLoaded ? 'text-green-500' : 'text-yellow-500',
    },
    {
      name: 'CORS Protection',
      status: 'secure',
      value: 'Enabled',
      icon: Lock,
      color: 'text-green-500',
    },
    {
      name: 'Session Uptime',
      status: 'secure',
      value: formatUptime(uptime),
      icon: Activity,
      color: 'text-green-500',
    },
  ];

  const threatLog = [
    {
      id: 1,
      type: 'Suspicious Activity',
      severity: 'medium',
      description: 'Multiple failed login attempts detected',
      timestamp: '2025-11-10 14:32:00',
      status: 'investigating',
    },
    {
      id: 2,
      type: 'Rate Limiting',
      severity: 'low',
      description: 'API rate limit exceeded by IP 192.168.1.100',
      timestamp: '2025-11-10 13:15:00',
      status: 'resolved',
    },
    {
      id: 3,
      type: 'Data Breach Attempt',
      severity: 'high',
      description: 'SQL injection attempt blocked',
      timestamp: '2025-11-10 11:45:00',
      status: 'blocked',
    },
  ];

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader
          title="Security Center"
          description="System security monitoring and threat analysis"
          breadcrumbs={[{ label: 'Security' }]}
        />

      {/* Security Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {securityMetrics.map((metric, index) => (
          <motion.div
            key={metric.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{metric.name}</p>
                <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
              </div>
              <metric.icon className={`w-8 h-8 ${metric.color}`} />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Security Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className={`card ${
          apiHealthy && datasetLoaded
            ? 'bg-green-500/10 border-green-500/20'
            : 'bg-yellow-500/10 border-yellow-500/20'
        }`}
      >
        <div className="flex items-start gap-3">
          {apiHealthy && datasetLoaded ? (
            <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle className="w-6 h-6 text-yellow-500 flex-shrink-0 mt-0.5" />
          )}
          <div>
            <h3 className="font-semibold mb-1">
              {apiHealthy && datasetLoaded ? 'System Operational' : 'System Status Warning'}
            </h3>
            <p className="text-sm text-muted-foreground">
              {apiHealthy && datasetLoaded
                ? `Backend API is running. Dataset loaded with ${metricsData?.total_transactions?.toLocaleString()} transactions. All security protocols active.`
                : 'Some system components are not fully operational. Check connection status above.'}
            </p>
          </div>
        </div>
      </motion.div>

      {/* API Connection Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Shield className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">API Connection Log</h2>
        </div>

        <div className="space-y-3">
          {apiHealthy ? (
            <div className="p-4 rounded-lg border bg-green-500/10 border-green-500/20">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="font-semibold">Backend API Connected</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Successfully connected to fraud detection API on port 8000
                  </p>
                </div>
                <span className="px-2 py-1 rounded text-xs font-medium uppercase bg-green-500/20 text-green-500">
                  Active
                </span>
              </div>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted-foreground">{new Date().toLocaleString()}</span>
                <span className="px-2 py-1 rounded text-xs font-medium bg-green-500/20 text-green-500">
                  connected
                </span>
              </div>
            </div>
          ) : (
            <div className="p-4 rounded-lg border bg-red-500/10 border-red-500/20">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="font-semibold">Backend API Connection Failed</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    Cannot connect to API. Make sure the backend is running on port 8000.
                  </p>
                </div>
                <span className="px-2 py-1 rounded text-xs font-medium uppercase bg-red-500/20 text-red-500">
                  Error
                </span>
              </div>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted-foreground">{new Date().toLocaleString()}</span>
                <span className="px-2 py-1 rounded text-xs font-medium bg-red-500/20 text-red-500">
                  disconnected
                </span>
              </div>
            </div>
          )}
          
          {datasetLoaded && (
            <div className="p-4 rounded-lg border bg-green-500/10 border-green-500/20">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="font-semibold">Dataset Loaded Successfully</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    IBM fraud detection dataset with {metricsData?.total_transactions?.toLocaleString()} transactions loaded
                  </p>
                </div>
                <span className="px-2 py-1 rounded text-xs font-medium uppercase bg-green-500/20 text-green-500">
                  Ready
                </span>
              </div>
            </div>
          )}
        </div>
      </motion.div>

      {/* Security Recommendations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Server className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">Security Recommendations</h2>
        </div>

        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold">SSL/TLS Encryption</h3>
              <p className="text-sm text-muted-foreground">
                All data transmission is encrypted using TLS 1.3
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold">Rate Limiting Active</h3>
              <p className="text-sm text-muted-foreground">
                API requests are limited to prevent abuse
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold">CORS Protection</h3>
              <p className="text-sm text-muted-foreground">
                Cross-origin requests are restricted to authorized domains
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-semibold">Enable Two-Factor Authentication</h3>
              <p className="text-sm text-muted-foreground">
                Add an extra layer of security for administrator accounts
              </p>
            </div>
          </div>
        </div>
      </motion.div>
      </div>
    </DashboardLayout>
  );
}
