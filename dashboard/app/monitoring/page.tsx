'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Pause, Download } from 'lucide-react';
import useSWR from 'swr';
import { api, Transaction } from '@/lib/api';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';

export default function MonitoringPage() {
  const [isPaused, setIsPaused] = useState(false);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filter, setFilter] = useState<'all' | 'fraud' | 'safe'>('all');

  const { data } = useSWR(
    '/api/transactions/recent',
    () => api.getRecentTransactions(50),
    { refreshInterval: isPaused ? 0 : 2000 }
  );

  useEffect(() => {
    if (data?.transactions) {
      setTransactions(prev => {
        const newTxns = data.transactions.filter(
          (t: Transaction) => !prev.find(p => p.id === t.id)
        );
        return [...newTxns, ...prev].slice(0, 100);
      });
    }
  }, [data]);

  const filteredTransactions = transactions.filter(t => {
    if (filter === 'fraud') return t.risk === 'high' || t.risk === 'medium';
    if (filter === 'safe') return t.risk === 'low';
    return true;
  });

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'text-red-500 bg-red-500/10 border-red-500/30';
      case 'medium': return 'text-yellow-500 bg-yellow-500/10 border-yellow-500/30';
      case 'low': return 'text-green-500 bg-green-500/10 border-green-500/30';
      default: return 'text-gray-500 bg-gray-500/10 border-gray-500/30';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'blocked': return 'text-red-500';
      case 'flagged': return 'text-yellow-500';
      case 'reviewing': return 'text-blue-500';
      case 'approved': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  const actions = (
    <div className="flex items-center gap-2">
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => setIsPaused(!isPaused)}
        className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
          isPaused ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
        }`}
      >
        {isPaused ? <><Play className="w-4 h-4" /> Resume</> : <><Pause className="w-4 h-4" /> Pause</>}
      </motion.button>
      <motion.button
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={() => alert('Export functionality coming soon!')}
        className="px-4 py-2 bg-accent hover:bg-accent/80 rounded-lg flex items-center gap-2"
      >
        <Download className="w-4 h-4" /> Export
      </motion.button>
    </div>
  );

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader title="Live Monitoring" description="Real-time transaction stream with fraud detection" breadcrumbs={[{ label: 'Live Monitoring' }]} actions={actions} />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card">
          <h3 className="text-sm text-muted-foreground mb-1">Total</h3>
          <p className="text-3xl font-bold">{transactions.length}</p>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card">
          <h3 className="text-sm text-muted-foreground mb-1">High Risk</h3>
          <p className="text-3xl font-bold text-red-500">{transactions.filter(t => t.risk === 'high').length}</p>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card">
          <h3 className="text-sm text-muted-foreground mb-1">Medium Risk</h3>
          <p className="text-3xl font-bold text-yellow-500">{transactions.filter(t => t.risk === 'medium').length}</p>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card">
          <h3 className="text-sm text-muted-foreground mb-1">Safe</h3>
          <p className="text-3xl font-bold text-green-500">{transactions.filter(t => t.risk === 'low').length}</p>
        </motion.div>
      </div>
      <div className="flex gap-2">
        <button onClick={() => setFilter('all')} className={`px-4 py-2 rounded-lg text-sm ${filter === 'all' ? 'bg-primary text-primary-foreground' : 'bg-accent'}`}>All</button>
        <button onClick={() => setFilter('fraud')} className={`px-4 py-2 rounded-lg text-sm ${filter === 'fraud' ? 'bg-red-500 text-white' : 'bg-accent'}`}>Fraud</button>
        <button onClick={() => setFilter('safe')} className={`px-4 py-2 rounded-lg text-sm ${filter === 'safe' ? 'bg-green-500 text-white' : 'bg-accent'}`}>Safe</button>
      </div>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Transaction Stream</h2>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm text-green-500">{isPaused ? 'Paused' : 'Live'}</span>
          </div>
        </div>
        <div className="space-y-2 max-h-[600px] overflow-y-auto">
          <AnimatePresence>
            {filteredTransactions.map((tx) => (
              <motion.div key={tx.id} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }} className="flex items-center gap-4 p-4 rounded-lg border border-border hover:border-primary/50 transition-all">
                <div className="text-2xl"></div>
                <div className="w-32"><p className="font-bold text-lg">${tx.amount.toLocaleString()}</p></div>
                <div className="flex-1 font-mono text-sm">{tx.source}  {tx.target}</div>
                <div className={`px-3 py-1.5 rounded-full border ${getRiskColor(tx.risk)}`}><span className="text-xs font-semibold uppercase">{tx.risk}</span></div>
                <div className={`capitalize ${getStatusColor(tx.status)}`}>{tx.status}</div>
                <div className="text-sm text-muted-foreground w-24 text-right">{new Date(tx.timestamp).toLocaleTimeString()}</div>
              </motion.div>
            ))}
          </AnimatePresence>
          {filteredTransactions.length === 0 && <div className="text-center py-12 text-muted-foreground"><p>No transactions</p></div>}
        </div>
      </motion.div>
      </div>
    </DashboardLayout>
  );
}
