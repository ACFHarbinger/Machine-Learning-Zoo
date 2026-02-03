/**
 * DataNode - Custom React Flow node for data input/output
 */
import { Handle, Position, type NodeProps } from 'reactflow';

interface DataNodeData {
  label: string;
  nodeKind?: 'input' | 'output';
}

export default function DataNode({ data, selected }: NodeProps<DataNodeData>) {
  const isInput = data.nodeKind !== 'output';

  return (
    <div className={`min-w-[140px] rounded-xl border-2 overflow-hidden transition-all hover:scale-105 ${
      selected 
        ? 'border-white/50 shadow-xl' 
        : isInput 
          ? 'border-cyan-500/40' 
          : 'border-rose-500/40'
    } ${isInput ? 'bg-cyan-950/80' : 'bg-rose-950/80'}`}>
      {/* Header */}
      <div className={`h-2 ${isInput ? 'bg-gradient-to-r from-cyan-500 to-blue-500' : 'bg-gradient-to-r from-rose-500 to-pink-500'}`} />
      
      {/* Content */}
      <div className="px-4 py-3 flex items-center gap-2">
        {isInput ? (
          <svg className="w-4 h-4 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
        ) : (
          <svg className="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
        )}
        <span className={`font-semibold text-sm ${isInput ? 'text-cyan-300' : 'text-rose-300'}`}>
          {data.label || (isInput ? 'Data Input' : 'Output')}
        </span>
      </div>

      {/* Handles */}
      {!isInput && (
        <Handle
          type="target"
          position={Position.Left}
          className="!w-3 !h-3 !bg-slate-700 !border-2 !border-rose-500"
        />
      )}
      {isInput && (
        <Handle
          type="source"
          position={Position.Right}
          className="!w-3 !h-3 !bg-cyan-500 !border-2 !border-cyan-400"
        />
      )}
    </div>
  );
}
