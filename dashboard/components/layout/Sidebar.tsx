'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Activity,
  Network,
  BarChart3,
  Settings,
  Bell,
  TrendingUp,
  Shield,
} from 'lucide-react';

interface SidebarProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const navItems = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Live Monitoring', href: '/monitoring', icon: Activity },
  { name: 'Graph Visualization', href: '/graph', icon: Network },
  { name: 'Model Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'Experiments', href: '/experiments', icon: TrendingUp },
  { name: 'Alerts', href: '/alerts', icon: Bell },
  { name: 'Security', href: '/security', icon: Shield },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Sidebar({ open, setOpen }: SidebarProps) {
  const pathname = usePathname();

  return (
    <motion.aside
      initial={false}
      animate={{ width: open ? 256 : 80 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="relative bg-card border-r border-border flex flex-col"
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-center border-b border-border">
        <motion.div
          animate={{ scale: open ? 1 : 0.8 }}
          transition={{ duration: 0.2 }}
        >
          {open ? (
            <h2 className="text-xl font-bold gradient-text">TGNN Monitor</h2>
          ) : (
            <Shield className="w-8 h-8 text-primary" />
          )}
        </motion.div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link key={item.name} href={item.href}>
              <motion.div
                whileHover={{ scale: 1.05, x: 5 }}
                whileTap={{ scale: 0.95 }}
                className={`
                  flex items-center gap-4 px-4 py-3 rounded-lg cursor-pointer
                  transition-colors duration-200
                  ${
                    isActive
                      ? 'bg-primary text-primary-foreground glow-blue'
                      : 'hover:bg-accent text-muted-foreground hover:text-foreground'
                  }
                `}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                {open && (
                  <motion.span
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="font-medium"
                  >
                    {item.name}
                  </motion.span>
                )}
              </motion.div>
            </Link>
          );
        })}
      </nav>

      {/* Status Indicator */}
      {open && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 border-t border-border"
        >
          <div className="flex items-center gap-3 p-3 bg-accent/50 rounded-lg">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <div className="flex-1">
              <p className="text-xs font-medium">System Status</p>
              <p className="text-xs text-muted-foreground">All systems operational</p>
            </div>
          </div>
        </motion.div>
      )}
    </motion.aside>
  );
}
