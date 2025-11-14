'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { GraphStructure } from '@/lib/api';

interface Graph3DProps {
  data: GraphStructure;
  isRotating: boolean;
  selectedNode: number | null;
  onNodeClick: (nodeId: number) => void;
}

function Node({ 
  position, 
  color, 
  size, 
  onClick,
  isSelected 
}: { 
  position: [number, number, number]; 
  color: string; 
  size: number;
  onClick: () => void;
  isSelected: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current && isSelected) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.2);
    }
  });

  return (
    <mesh ref={meshRef} position={position} onClick={onClick}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial 
        color={color} 
        emissive={color}
        emissiveIntensity={isSelected ? 0.5 : 0.2}
        metalness={0.3}
        roughness={0.4}
      />
    </mesh>
  );
}

function Edge({ 
  start, 
  end, 
  color 
}: { 
  start: [number, number, number]; 
  end: [number, number, number]; 
  color: string;
}) {
  const points = useMemo(() => {
    return [new THREE.Vector3(...start), new THREE.Vector3(...end)];
  }, [start, end]);

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length}
          array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} opacity={0.3} transparent />
    </line>
  );
}

function Scene({ data, isRotating, selectedNode, onNodeClick }: Graph3DProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Calculate 3D positions for nodes using force-directed layout
  const nodePositions = useMemo(() => {
    const positions = new Map<number, [number, number, number]>();
    const numNodes = data.nodes.length;
    
    // Create a sphere layout for better 3D distribution
    data.nodes.forEach((node, i) => {
      const phi = Math.acos(-1 + (2 * i) / numNodes);
      const theta = Math.sqrt(numNodes * Math.PI) * phi;
      const radius = 15;
      
      positions.set(node.id, [
        radius * Math.cos(theta) * Math.sin(phi),
        radius * Math.sin(theta) * Math.sin(phi),
        radius * Math.cos(phi),
      ]);
    });
    
    return positions;
  }, [data.nodes]);

  useFrame((state, delta) => {
    if (groupRef.current && isRotating) {
      groupRef.current.rotation.y += delta * 0.2;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Render edges */}
      {data.edges.map((edge, i) => {
        const startPos = nodePositions.get(edge.source);
        const endPos = nodePositions.get(edge.target);
        
        if (!startPos || !endPos) return null;
        
        return (
          <Edge
            key={`edge-${i}`}
            start={startPos}
            end={endPos}
            color={edge.fraud ? '#ef4444' : '#6b7280'}
          />
        );
      })}

      {/* Render nodes */}
      {data.nodes.map((node) => {
        const pos = nodePositions.get(node.id);
        if (!pos) return null;

        const color = node.fraud ? '#ef4444' : '#10b981';
        const size = 0.3;

        return (
          <Node
            key={`node-${node.id}`}
            position={pos}
            color={color}
            size={size}
            onClick={() => onNodeClick(node.id)}
            isSelected={selectedNode === node.id}
          />
        );
      })}

      {/* Ambient light */}
      <ambientLight intensity={0.5} />
      
      {/* Directional lights */}
      <directionalLight position={[10, 10, 10]} intensity={1} />
      <directionalLight position={[-10, -10, -10]} intensity={0.5} />
      
      {/* Point light for dramatic effect */}
      <pointLight position={[0, 0, 0]} intensity={0.5} color="#4a90e2" />
    </group>
  );
}

export default function Graph3D({ data, isRotating, selectedNode, onNodeClick }: Graph3DProps) {
  return (
    <Canvas
      camera={{ position: [0, 0, 30], fov: 75 }}
      style={{ background: 'linear-gradient(to bottom, #0f172a, #1e293b)' }}
    >
      <Scene 
        data={data} 
        isRotating={isRotating} 
        selectedNode={selectedNode}
        onNodeClick={onNodeClick}
      />
      <OrbitControls 
        enableZoom={true}
        enablePan={true}
        enableRotate={true}
        minDistance={10}
        maxDistance={100}
      />
    </Canvas>
  );
}
