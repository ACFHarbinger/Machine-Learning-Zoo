/**
 * ContainerNode - Node that can contain other modules (SkipConnection, HyperConnection)
 * 
 * Usage: Drag canvas nodes onto this container. When you release, overlapping nodes are absorbed.
 */
import { memo, useCallback } from 'react';
import { Handle, Position, type NodeProps, useNodeId, useReactFlow, type Node } from 'reactflow';

interface ContainerNodeData {
  label: string;
  category?: string;
  modelType?: string;
  params?: Record<string, unknown>;
  children?: Array<{
    id: string;
    name: string;
    params: Record<string, unknown>;
  }>;
}

function ContainerNode({ data, selected }: NodeProps<ContainerNodeData>) {
  const nodeId = useNodeId();
  const { setNodes } = useReactFlow();
  
  // Use data.children directly from React Flow state
  const children = data.children || [];

  const removeChild = useCallback((childId: string) => {
    if (!nodeId) return;
    
    const child = children.find(c => c.id === childId);
    if (!child) return;

    // Remove from children and add back to canvas
    setNodes((nodes) => {
      const thisNode = nodes.find(n => n.id === nodeId);
      if (!thisNode) return nodes;

      // Create new node on canvas from child
      const newNode: Node = {
        id: child.id,
        type: 'model',
        position: {
          x: thisNode.position.x + 20,
          y: thisNode.position.y + 100,
        },
        data: {
          label: child.name,
          params: child.params,
        },
      };

      // Update container to remove child
      const updatedNodes = nodes.map(n => 
        n.id === nodeId 
          ? { ...n, data: { ...n.data, children: children.filter(c => c.id !== childId) } }
          : n
      );

      return [...updatedNodes, newNode];
    });
  }, [nodeId, children, setNodes]);

  return (
    <div
      className={`min-w-[260px] rounded-xl shadow-2xl transition-all duration-200 ${
        selected 
          ? 'ring-2 ring-violet-500 shadow-violet-500/20' 
          : 'hover:shadow-violet-500/10'
      }`}
    >
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-violet-500 !border-2 !border-slate-900"
      />

      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-violet-600 to-purple-600 rounded-t-xl">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-white/80" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
          <span className="text-sm font-semibold text-white">{data.label}</span>
        </div>
        <div className="text-xs text-white/60 mt-0.5">Container • {data.category}</div>
      </div>

      {/* Content Area */}
      <div className="p-3 bg-slate-800 rounded-b-xl min-h-[100px]">
        {children && children.length > 0 ? (
          <div className="space-y-2">
            {children.map((child) => (
              <div
                key={child.id}
                className="flex items-center justify-between px-3 py-2 bg-slate-700/50 rounded-lg border border-slate-600 hover:border-slate-500 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-indigo-500" />
                  <span className="text-xs text-slate-300 font-medium">{child.name}</span>
                </div>
                <button
                  onClick={() => removeChild(child.id)}
                  className="p-1 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
                  title="Remove from container"
                >
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-6 text-center text-slate-500">
            <svg className="w-8 h-8 mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span className="text-xs font-medium">Drag nodes here</span>
            <span className="text-[10px] mt-1 opacity-70">Drop canvas nodes to nest</span>
          </div>
        )}
      </div>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-violet-500 !border-2 !border-slate-900"
      />
    </div>
  );
}

export default memo(ContainerNode);
