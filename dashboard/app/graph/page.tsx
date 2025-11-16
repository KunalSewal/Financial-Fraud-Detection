'use client';

import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import useSWR from 'swr';
import dynamic from 'next/dynamic';
import DashboardLayout from '@/components/layout/DashboardLayout';
import PageHeader from '@/components/layout/PageHeader';
import { api, GraphStructure, EgoNetwork, CommunitiesResponse } from '@/lib/api';
import {
  Network,
  Users,
  TrendingUp,
  AlertCircle,
  Search,
  Loader2,
} from 'lucide-react';

// Dynamically import ForceGraph2D (client-side only)
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full">
      <Loader2 className="w-8 h-8 animate-spin text-primary" />
    </div>
  ),
});

type ViewMode = 'overview' | 'communities' | 'ego';

export default function GraphVisualization() {
  const [viewMode, setViewMode] = useState<ViewMode>('overview');
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [sampleSize, setSampleSize] = useState(200);
  const [searchNodeId, setSearchNodeId] = useState('');
  const graphRef = useRef<any>();

  // Fetch data based on view mode
  const { data: graphData, isLoading: graphLoading } = useSWR<GraphStructure>(
    viewMode === 'overview' ? `/api/graph/structure?sample_size=${sampleSize}` : null,
    () => api.getGraphStructure(sampleSize),
    { refreshInterval: 0 }
  );

  const { data: communities, isLoading: communitiesLoading } = useSWR<CommunitiesResponse>(
    viewMode === 'communities' ? '/api/graph/communities' : null,
    () => api.getFraudCommunities(1000),
    { refreshInterval: 0 }
  );

  const { data: egoNetwork, isLoading: egoLoading } = useSWR<EgoNetwork | null>(
    viewMode === 'ego' && selectedNode ? `/api/graph/ego-network?node=${selectedNode}` : null,
    async () => {
      if (!selectedNode) return null;
      return api.getEgoNetwork(selectedNode, 2);
    },
    { refreshInterval: 0 }
  );

  // Prepare graph data for ForceGraph2D
  const prepareGraphData = () => {
    if (viewMode === 'overview' && graphData) {
      return {
        nodes: graphData.nodes.map(n => ({
          id: n.id,
          name: n.label,
          fraud: n.fraud,
          val: n.fraud ? 5 : 3,
        })),
        links: graphData.edges.map(e => ({
          source: e.source,
          target: e.target,
          fraud: e.fraud,
        })),
      };
    }

    if (viewMode === 'ego' && egoNetwork) {
      return {
        nodes: egoNetwork.nodes.map(n => ({
          id: n.id,
          name: n.label,
          fraud: n.fraud,
          is_center: n.is_center,
          val: n.is_center ? 10 : (n.fraud ? 5 : 3),
        })),
        links: egoNetwork.edges.map(e => ({
          source: e.source,
          target: e.target,
        })),
      };
    }

    return { nodes: [], links: [] };
  };

  const graphDataViz = prepareGraphData();

  // Calculate stats
  const stats = {
    totalNodes: graphDataViz.nodes.length,
    totalEdges: graphDataViz.links.length,
    fraudNodes: graphDataViz.nodes.filter((n: any) => n.fraud).length,
  };

  const handleNodeClick = (node: any) => {
    setSelectedNode(node.id);
    if (viewMode === 'overview') {
      setViewMode('ego');
    }
  };

  const handleSearchNode = () => {
    const nodeId = parseInt(searchNodeId);
    if (!isNaN(nodeId)) {
      setSelectedNode(nodeId);
      setViewMode('ego');
    }
  };

  const isLoading = graphLoading || communitiesLoading || egoLoading;

  return (
    <DashboardLayout>
      <div className="space-y-8">
        <PageHeader
          title="Network Analysis"
          description="2D force-directed graph with fraud community detection"
        />

      {/* View Mode Selector */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { id: 'overview', label: 'Graph Overview', icon: Network, desc: '2D Force-Directed Layout' },
          { id: 'communities', label: 'Fraud Communities', icon: Users, desc: 'Detect Fraud Clusters' },
          { id: 'ego', label: 'Ego Network', icon: Search, desc: 'Node Neighborhood Analysis' },
        ].map((mode) => (
          <motion.button
            key={mode.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setViewMode(mode.id as ViewMode)}
            className={`p-4 rounded-lg border-2 transition-all ${
              viewMode === mode.id
                ? 'border-primary bg-primary/10'
                : 'border-border bg-card hover:border-primary/50'
            }`}
          >
            <mode.icon className={`w-6 h-6 mx-auto mb-2 ${viewMode === mode.id ? 'text-primary' : 'text-muted-foreground'}`} />
            <div className="text-sm font-semibold">{mode.label}</div>
            <div className="text-xs text-muted-foreground mt-1">{mode.desc}</div>
          </motion.button>
        ))}
      </div>

      {/* Search Bar (for Ego mode) */}
      {viewMode === 'ego' && (
        <div className="flex gap-2">
          <input
            type="number"
            value={searchNodeId}
            onChange={(e) => setSearchNodeId(e.target.value)}
            placeholder="Enter node ID to analyze..."
            className="flex-1 px-4 py-2 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
          />
          <button
            onClick={handleSearchNode}
            className="px-6 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors flex items-center gap-2"
          >
            <Search className="w-4 h-4" />
            Analyze
          </button>
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 bg-card rounded-lg border border-border"
        >
          <div className="text-sm text-muted-foreground mb-1">Total Nodes</div>
          <div className="text-2xl font-bold">{stats.totalNodes.toLocaleString()}</div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="p-4 bg-card rounded-lg border border-border"
        >
          <div className="text-sm text-muted-foreground mb-1">Total Edges</div>
          <div className="text-2xl font-bold">{stats.totalEdges.toLocaleString()}</div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="p-4 bg-card rounded-lg border border-border"
        >
          <div className="text-sm text-muted-foreground mb-1">Fraud Nodes</div>
          <div className="text-2xl font-bold text-red-500">{stats.fraudNodes.toLocaleString()}</div>
        </motion.div>
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-3 gap-6">
        {/* Graph Visualization */}
        <div className="col-span-2">
          <div className="bg-card rounded-lg border border-border overflow-hidden" style={{ height: '600px' }}>
            {isLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
                  <p className="text-muted-foreground">Loading network data...</p>
                </div>
              </div>
            ) : graphDataViz.nodes.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <AlertCircle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">No data available</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    {viewMode === 'ego' ? 'Enter a node ID to analyze its neighborhood' : 'Try adjusting sample size or switching datasets'}
                  </p>
                </div>
              </div>
            ) : (
              <ForceGraph2D
                ref={graphRef}
                graphData={graphDataViz}
                nodeLabel="name"
                nodeColor={(node: any) => 
                  node.is_center ? '#3b82f6' : (node.fraud ? '#ef4444' : '#10b981')
                }
                nodeRelSize={6}
                nodeVal={(node: any) => node.val || 3}
                linkColor={(link: any) => (link.fraud ? '#ef4444' : '#6b7280')}
                linkWidth={2}
                onNodeClick={handleNodeClick}
                backgroundColor="#0f172a"
                cooldownTicks={100}
                d3AlphaDecay={0.02}
                d3VelocityDecay={0.3}
              />
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {viewMode === 'communities' && communities && (
            <div className="bg-card rounded-lg border border-border p-4">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Users className="w-5 h-5" />
                Fraud Communities
              </h3>
              <div className="space-y-2 max-h-[500px] overflow-y-auto">
                {communities.communities.map((community) => (
                  <motion.div
                    key={community.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: community.id * 0.05 }}
                    className="p-3 bg-background rounded-lg border border-border hover:border-primary cursor-pointer transition-colors"
                    onClick={() => {
                      setSelectedNode(community.nodes[0]);
                      setViewMode('ego');
                    }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold">Community #{community.id + 1}</span>
                      <span className="text-xs bg-red-500/20 text-red-500 px-2 py-1 rounded">
                        {community.size} nodes
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Click to explore this fraud cluster
                    </div>
                  </motion.div>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-border text-sm text-muted-foreground">
                <div>Total Communities: {communities.total_communities}</div>
                <div>Total Fraud Nodes: {communities.total_fraud_nodes}</div>
              </div>
            </div>
          )}

          {viewMode === 'overview' && (
            <div className="bg-card rounded-lg border border-border p-4">
              <h3 className="text-lg font-semibold mb-4">Controls</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-muted-foreground mb-2 block">
                    Sample Size: {sampleSize} nodes
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="500"
                    step="50"
                    value={sampleSize}
                    onChange={(e) => setSampleSize(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>50 (sparse)</span>
                    <span>500 (dense)</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {selectedNode && viewMode === 'ego' && egoNetwork && (
            <div className="bg-card rounded-lg border border-border p-4">
              <h3 className="text-lg font-semibold mb-4">Selected Node</h3>
              <div className="space-y-3">
                <div>
                  <div className="text-sm text-muted-foreground">Node ID</div>
                  <div className="text-xl font-bold">{selectedNode}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Neighbors</div>
                  <div className="text-xl font-bold">{egoNetwork.nodes.length - 1}</div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">Connections</div>
                  <div className="text-xl font-bold">{egoNetwork.edges.length}</div>
                </div>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="bg-card rounded-lg border border-border p-4">
            <h3 className="text-sm font-semibold mb-3">Legend</h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <span>Fraud Node</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span>Normal Node</span>
              </div>
              {viewMode === 'ego' && (
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-blue-500" />
                  <span>Center Node</span>
                </div>
              )}
              <div className="pt-2 mt-2 border-t border-border text-muted-foreground">
                Click nodes to explore their neighborhood
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>
    </DashboardLayout>
  );
}
