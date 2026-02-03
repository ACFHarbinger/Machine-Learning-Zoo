/**
 * PropertiesPanel - Edit selected node parameters
 */
import type { Node } from 'reactflow';

interface Props {
  selectedNode: Node | null;
  onClose: () => void;
  onUpdateData: (nodeId: string, data: Record<string, unknown>) => void;
  onRemove: (nodeId: string) => void;
}

export default function PropertiesPanel({ selectedNode, onClose, onUpdateData, onRemove }: Props) {
  if (!selectedNode) {
    return (
      <div className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col items-center justify-center text-center p-8">
        <div className="w-16 h-16 rounded-2xl bg-slate-800 flex items-center justify-center mb-4">
          <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <h3 className="text-slate-400 font-medium mb-2">No Node Selected</h3>
        <p className="text-sm text-slate-600">Click on a node in the canvas to view and edit its properties.</p>
      </div>
    );
  }

  const data = selectedNode.data as Record<string, unknown>;
  const params = (data.params || {}) as Record<string, unknown>;

  const updateParam = (key: string, value: unknown) => {
    onUpdateData(selectedNode.id, { params: { ...params, [key]: value } });
  };

  const getInputType = (value: unknown) => {
    if (typeof value === 'number') return 'number';
    if (typeof value === 'boolean') return 'checkbox';
    return 'text';
  };

  return (
    <div className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${
            selectedNode.type === 'data' ? 'bg-cyan-500' :
            data.modelType === 'mac' ? 'bg-emerald-500' : 'bg-indigo-500'
          }`} />
          <div>
            <h2 className="font-semibold text-white">{String(data.label || 'Unnamed')}</h2>
            <p className="text-xs text-slate-500">{String(data.category || selectedNode.type)}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 text-slate-500 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Parameters */}
      <div className="flex-1 overflow-y-auto p-4">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4">Parameters</h3>
        
        {Object.entries(params).length === 0 ? (
          <p className="text-sm text-slate-600">No configurable parameters.</p>
        ) : (
          <div className="space-y-4">
            {Object.entries(params).map(([key, value]) => (
              <div key={key}>
                <label className="block text-sm text-slate-400 mb-1.5">{key.replace(/_/g, ' ')}</label>
                {typeof value === 'boolean' ? (
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={(e) => updateParam(key, e.target.checked)}
                      className="w-5 h-5 rounded border-slate-700 bg-slate-800 text-indigo-500 focus:ring-indigo-500 focus:ring-offset-0"
                    />
                    <span className="text-sm text-slate-300">{value ? 'Enabled' : 'Disabled'}</span>
                  </label>
                ) : (
                  <input
                    type={getInputType(value)}
                    value={String(value)}
                    onChange={(e) => updateParam(key, getInputType(value) === 'number' ? Number(e.target.value) : e.target.value)}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t border-slate-800">
        <button
          onClick={() => onRemove(selectedNode.id)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/30 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors text-sm"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          Remove Node
        </button>
      </div>
    </div>
  );
}
