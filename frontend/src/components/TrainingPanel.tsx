/**
 * TrainingPanel - Training configuration and controls
 */
import { useState } from 'react';
import type { Node } from 'reactflow';

const TRAINING_MODES = [
  { id: 'supervised', label: 'Supervised' },
  { id: 'self_supervised', label: 'Self-Supervised' },
  { id: 'reinforcement', label: 'RL' },
  { id: 'federated', label: 'Federated' },
  { id: 'continual', label: 'Continual' },
  { id: 'transfer', label: 'Transfer' },
] as const;

const HPO_ALGORITHMS = [
  { id: 'none', label: 'None (Manual)' },
  { id: 'grid', label: 'Grid Search' },
  { id: 'random', label: 'Random Search' },
  { id: 'bayesian', label: 'Bayesian' },
  { id: 'dehb', label: 'DEHB' },
  { id: 'optuna', label: 'Optuna' },
];

interface Props {
  nodes: Node[];
}

export default function TrainingPanel({ nodes }: Props) {
  const [mode, setMode] = useState('supervised');
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [hpoAlgorithm, setHpoAlgorithm] = useState('none');
  const [isTraining, setIsTraining] = useState(false);

  const handleStartTraining = () => {
    if (nodes.length < 2) {
      alert('Add at least an input and output node to start training.');
      return;
    }
    setIsTraining(true);
    setTimeout(() => setIsTraining(false), 3000);
  };

  return (
    <div className="bg-slate-900/80 backdrop-blur border-t border-slate-800 px-6 py-4">
      <div className="flex items-start gap-8">
        {/* Training Mode */}
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Training Mode</h3>
          <div className="flex flex-wrap gap-2">
            {TRAINING_MODES.map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setMode(id)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  mode === id
                    ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40'
                    : 'bg-slate-800 text-slate-400 border border-transparent hover:bg-slate-700'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Hyperparameters */}
        <div className="flex gap-4">
          <div>
            <label className="block text-xs text-slate-500 mb-1.5">Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              className="w-24 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1.5">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-24 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-500 mb-1.5">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={learningRate}
              onChange={(e) => setLearningRate(Number(e.target.value))}
              className="w-28 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>

        {/* HPO */}
        <div>
          <label className="block text-xs text-slate-500 mb-1.5">HPO Algorithm</label>
          <select
            value={hpoAlgorithm}
            onChange={(e) => setHpoAlgorithm(e.target.value)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {HPO_ALGORITHMS.map(({ id, label }) => (
              <option key={id} value={id}>{label}</option>
            ))}
          </select>
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Actions */}
        <div className="flex items-center gap-3">
          <div className="text-sm text-slate-500">
            {nodes.length} nodes
          </div>
          <button
            onClick={handleStartTraining}
            disabled={isTraining}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold transition-all ${
              isTraining
                ? 'bg-indigo-500/50 text-indigo-200 cursor-wait'
                : 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:from-indigo-600 hover:to-purple-700 shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40'
            }`}
          >
            {isTraining ? (
              <>
                <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Training...
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Start Training
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
