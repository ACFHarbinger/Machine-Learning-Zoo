/**
 * ML Zoo Model Builder Dashboard
 * Premium drag-and-drop interface for building ML pipelines
 */
import { useCallback, useRef, useState } from 'react';
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
  useNodesState,
  useEdgesState,
  addEdge,
  type Node,
  type Connection,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { nodeTypes } from './nodes';
import NodePalette from './components/NodePalette';
import PropertiesPanel from './components/PropertiesPanel';
import TrainingPanel from './components/TrainingPanel';

// Icons (inline SVG for reliability)
const ActivityIcon = () => (
  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
);

const ExportIcon = () => (
  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

let nodeId = 0;
const getNodeId = () => `node_${nodeId++}`;

function ModelBuilderCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  
  const selectedNode = nodes.find((n) => n.id === selectedNodeId) || null;

  const onConnect = useCallback((params: Connection) => {
    setEdges((eds) => addEdge({ ...params, animated: true, style: { stroke: '#6366f1', strokeWidth: 2 } }, eds));
  }, [setEdges]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    const bounds = reactFlowWrapper.current?.getBoundingClientRect();
    const rawData = event.dataTransfer.getData('application/reactflow');
    if (!rawData || !bounds) return;

    const data = JSON.parse(rawData);
    const position = {
      x: event.clientX - bounds.left - 100,
      y: event.clientY - bounds.top - 40,
    };

    const newNode: Node = {
      id: getNodeId(),
      type: data.type === 'data' ? 'data' : 'model',
      position,
      data: {
        label: data.name || data.label,
        modelType: data.modelType,
        category: data.category,
        params: data.params || {},
        nodeKind: data.nodeKind,
      },
    };

    setNodes((nds) => [...nds, newNode]);
  }, [setNodes]);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeId(node.id);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  const updateNodeData = (nodeId: string, newData: Record<string, unknown>) => {
    setNodes((nds) => nds.map((n) => 
      n.id === nodeId ? { ...n, data: { ...n.data, ...newData } } : n
    ));
  };

  const removeNode = (nodeId: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== nodeId));
    setEdges((eds) => eds.filter((e) => e.source !== nodeId && e.target !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  };

  const handleExport = () => {
    const pipeline = { nodes, edges };
    const blob = new Blob([JSON.stringify(pipeline, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pipeline.json';
    a.click();
  };

  return (
    <div className="flex h-screen bg-slate-950 text-white overflow-hidden">
      {/* Left Sidebar: Node Palette */}
      <NodePalette />

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <ActivityIcon />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                ML Zoo Model Builder
              </h1>
              <p className="text-xs text-slate-500">Drag & Drop Pipeline Designer</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg text-sm text-slate-300 transition-all hover:border-slate-600"
            >
              <ExportIcon /> Export Pipeline
            </button>
          </div>
        </header>

        {/* Canvas + Training Panel */}
        <div className="flex-1 flex flex-col">
          {/* React Flow Canvas */}
          <div ref={reactFlowWrapper} className="flex-1 bg-slate-950">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onNodeClick={onNodeClick}
              onPaneClick={onPaneClick}
              nodeTypes={nodeTypes}
              fitView
              snapToGrid
              snapGrid={[20, 20]}
              defaultEdgeOptions={{ animated: true, style: { stroke: '#6366f1', strokeWidth: 2 } }}
            >
              <Background color="#334155" gap={20} size={1} />
              <Controls 
                className="!bg-slate-800 !border-slate-700 !rounded-xl !shadow-xl [&>button]:!bg-slate-800 [&>button]:!border-slate-700 [&>button]:!text-slate-400 [&>button:hover]:!bg-slate-700" 
              />
              <MiniMap
                className="!bg-slate-900 !border-slate-800 !rounded-xl"
                nodeColor={(node: Node) => {
                  if (node.type === 'data') return '#06b6d4';
                  const modelType = (node.data as Record<string, unknown>)?.modelType;
                  if (modelType === 'mac') return '#10b981';
                  return '#6366f1';
                }}
              />
            </ReactFlow>
          </div>

          {/* Training Panel */}
          <TrainingPanel nodes={nodes} />
        </div>
      </div>

      {/* Right Sidebar: Properties Panel */}
      <PropertiesPanel 
        selectedNode={selectedNode}
        onClose={() => setSelectedNodeId(null)}
        onUpdateData={updateNodeData}
        onRemove={removeNode}
      />
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <ModelBuilderCanvas />
    </ReactFlowProvider>
  );
}
