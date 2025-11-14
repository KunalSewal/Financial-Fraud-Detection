'use client';

import DashboardLayout from '@/components/layout/DashboardLayout';
import MetricsGrid from '@/components/dashboard/MetricsGrid';
import ModelComparison from '@/components/dashboard/ModelComparison';
import RecentDetections from '@/components/dashboard/RecentDetections';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <DashboardLayout>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="space-y-8"
      >
        {/* Welcome Section */}
        <div>
          <h1 className="text-4xl font-bold gradient-text mb-2">
            Fraud Detection Dashboard
          </h1>
          <p className="text-muted-foreground text-lg">
            Real-time monitoring with Temporal Graph Neural Networks
          </p>
        </div>

        {/* Metrics Overview */}
        <MetricsGrid />

        {/* Model Performance Comparison */}
        <ModelComparison />

        {/* Recent Fraud Detections */}
        <RecentDetections />
      </motion.div>
    </DashboardLayout>
  );
}
