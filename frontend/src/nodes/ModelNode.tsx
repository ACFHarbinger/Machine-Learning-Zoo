/**
 * ModelNode - Custom React Flow node for ML models
 */
import { Handle, Position, type NodeProps } from 'reactflow';

interface ModelData {
  label: string;
  modelType?: 'deep' | 'mac' | 'helper';
  category?: string;
  params?: Record<string, unknown>;
}

export default function ModelNode({ data, selected }: NodeProps<ModelData>) {
  const getGradient = () => {
    switch (data.modelType) {
      case 'mac': return 'from-emerald-500 to-teal-600';
      case 'helper': return 'from-amber-500 to-orange-600';
      default: return 'from-indigo-500 to-purple-600';
    }
  };

  const getBorderColor = () => {
    if (selected) return 'border-white/50 shadow-xl shadow-indigo-500/30';
    switch (data.modelType) {
      case 'mac': return 'border-emerald-500/30';
      case 'helper': return 'border-amber-500/30';
      default: return 'border-indigo-500/30';
    }
  };

  return (
    <div className={`min-w-[160px] bg-slate-900 rounded-xl border-2 ${getBorderColor()} overflow-hidden transition-all hover:scale-105`}>
      {/* Header */}
      <div className={`h-2 bg-gradient-to-r ${getGradient()}`} />
      
      {/* Content */}
      <div className="px-4 py-3">
        <div className="font-semibold text-white text-sm">{data.label}</div>
        {data.category && (
          <div className="text-xs text-slate-500 mt-0.5">{data.category}</div>
        )}
      </div>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-slate-700 !border-2 !border-slate-500"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-indigo-500 !border-2 !border-indigo-400"
      />
    </div>
  );
}
