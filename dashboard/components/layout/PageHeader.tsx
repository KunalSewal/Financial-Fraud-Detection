'use client';

import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Home } from 'lucide-react';
import { ReactNode } from 'react';

interface PageHeaderProps {
  title: string;
  description?: string;
  breadcrumbs?: { label: string; href?: string }[];
  actions?: ReactNode;
}

export default function PageHeader({
  title,
  description,
  breadcrumbs = [],
  actions,
}: PageHeaderProps) {
  const router = useRouter();

  return (
    <div className="mb-8">
      {/* Breadcrumbs */}
      {breadcrumbs.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 mb-4 text-sm text-muted-foreground"
        >
          <button
            onClick={() => router.push('/')}
            className="hover:text-foreground transition-colors flex items-center gap-1"
          >
            <Home className="w-4 h-4" />
            Home
          </button>
          {breadcrumbs.map((crumb, index) => (
            <div key={index} className="flex items-center gap-2">
              <span>/</span>
              {crumb.href ? (
                <button
                  onClick={() => router.push(crumb.href!)}
                  className="hover:text-foreground transition-colors"
                >
                  {crumb.label}
                </button>
              ) : (
                <span className="text-foreground font-medium">{crumb.label}</span>
              )}
            </div>
          ))}
        </motion.div>
      )}

      {/* Header Content */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4 flex-1">
          {/* Back Button */}
          <motion.button
            whileHover={{ scale: 1.1, x: -2 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => router.back()}
            className="mt-1 p-2 rounded-lg bg-accent hover:bg-accent/80 text-foreground transition-colors"
            title="Go back"
          >
            <ArrowLeft className="w-5 h-5" />
          </motion.button>

          {/* Title & Description */}
          <div className="flex-1">
            <motion.h1
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-3xl font-bold gradient-text mb-2"
            >
              {title}
            </motion.h1>
            {description && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.1 }}
                className="text-muted-foreground"
              >
                {description}
              </motion.p>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        {actions && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            {actions}
          </motion.div>
        )}
      </div>
    </div>
  );
}
