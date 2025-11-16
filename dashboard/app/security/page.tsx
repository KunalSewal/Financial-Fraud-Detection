'use client';

import { motion } from 'framer-motion';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { Shield, Lock, AlertTriangle, CheckCircle, Activity, Server } from 'lucide-react';

export default function Security() {
  const securityMetrics = [
    {
      name: 'System Integrity',
      status: 'secure',
      value: '98%',
      icon: Shield,
      color: 'text-green-500',
    },
    {
      name: 'API Security',
      status: 'secure',
      value: 'Enabled',
      icon: Lock,
      color: 'text-green-500',
    },
    {
      name: 'Active Threats',
      status: 'warning',
      value: '3',
      icon: AlertTriangle,
      color: 'text-yellow-500',
    },
    {
      name: 'Uptime',
      status: 'secure',
      value: '99.9%',
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
        className="card bg-green-500/10 border-green-500/20"
      >
        <div className="flex items-start gap-3">
          <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold mb-1">All Systems Secure</h3>
            <p className="text-sm text-muted-foreground">
              No critical security issues detected. The system is operating normally with
              all security protocols active.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Threat Log */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Shield className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold">Threat Log</h2>
        </div>

        <div className="space-y-3">
          {threatLog.map((threat, index) => (
            <motion.div
              key={threat.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 + index * 0.1 }}
              className={`p-4 rounded-lg border ${
                threat.severity === 'high'
                  ? 'bg-red-500/10 border-red-500/20'
                  : threat.severity === 'medium'
                  ? 'bg-yellow-500/10 border-yellow-500/20'
                  : 'bg-blue-500/10 border-blue-500/20'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="font-semibold">{threat.type}</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    {threat.description}
                  </p>
                </div>
                <span
                  className={`px-2 py-1 rounded text-xs font-medium uppercase ${
                    threat.severity === 'high'
                      ? 'bg-red-500/20 text-red-500'
                      : threat.severity === 'medium'
                      ? 'bg-yellow-500/20 text-yellow-500'
                      : 'bg-blue-500/20 text-blue-500'
                  }`}
                >
                  {threat.severity}
                </span>
              </div>
              <div className="flex items-center gap-4 text-sm">
                <span className="text-muted-foreground">{threat.timestamp}</span>
                <span
                  className={`px-2 py-1 rounded text-xs font-medium ${
                    threat.status === 'blocked'
                      ? 'bg-red-500/20 text-red-500'
                      : threat.status === 'resolved'
                      ? 'bg-green-500/20 text-green-500'
                      : 'bg-yellow-500/20 text-yellow-500'
                  }`}
                >
                  {threat.status}
                </span>
              </div>
            </motion.div>
          ))}
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
