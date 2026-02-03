/**
 * NodePalette - Sidebar with draggable model blocks
 */
import { useState } from 'react';

// Model catalog
const MODEL_CATALOG: Record<string, Array<{ name: string; description: string; params: Record<string, unknown> }>> = {
  // Deep Learning
  'Recurrent Networks': [
    { name: 'LSTM', description: 'Long Short-Term Memory', params: { hidden_dim: 128, n_layers: 2, dropout: 0.1 } },
    { name: 'GRU', description: 'Gated Recurrent Unit', params: { hidden_dim: 128, n_layers: 2, dropout: 0.1 } },
    { name: 'xLSTM', description: 'Extended LSTM', params: { hidden_dim: 128, n_layers: 2, num_heads: 4 } },
    { name: 'Mamba', description: 'State-space model', params: { hidden_dim: 128, n_layers: 4, forecast_horizon: 1 } },
    { name: 'ESN', description: 'Echo State Network', params: { reservoir_dim: 500, spectral_radius: 0.9 } },
    { name: 'LSM', description: 'Liquid State Machine', params: { liquid_size: 1000 } },
  ],
  'Transformers': [
    { name: 'NSTransformer', description: 'Non-Stationary Transformer', params: { d_model: 128, n_heads: 8, seq_len: 96 } },
    { name: 'Attention', description: 'Multi-Head Attention', params: { d_model: 128, num_heads: 8 } },
  ],
  'Convolutional': [
    { name: 'CNN', description: 'Convolutional Network', params: { hidden_dim: 64, seq_len: 10 } },
    { name: 'ResNet', description: 'Residual Network', params: { hidden_dim: 64, num_blocks: 4 } },
    { name: 'TCN', description: 'Temporal Conv Net', params: { hidden_dim: 64, kernel_size: 3 } },
    { name: 'Capsule', description: 'Capsule Network', params: { in_caps: 8, out_caps: 10 } },
  ],
  'Autoencoders': [
    { name: 'AE', description: 'AutoEncoder', params: { latent_dim: 32 } },
    { name: 'VAE', description: 'Variational AE', params: { latent_dim: 32 } },
    { name: 'DAE', description: 'Denoising AE', params: { latent_dim: 32, noise: 0.1 } },
  ],
  'Spiking': [
    { name: 'SNN', description: 'Spiking Neural Network', params: { hidden_dim: 128, threshold: 1.0 } },
  ],
  'Memory Networks': [
    { name: 'NTM', description: 'Neural Turing Machine', params: { memory_size: 128, memory_dim: 20 } },
    { name: 'DNC', description: 'Differentiable Neural Computer', params: { memory_size: 128, num_reads: 4 } },
  ],
  'Probabilistic': [
    { name: 'RBM', description: 'Restricted Boltzmann Machine', params: { hidden_dim: 64 } },
    { name: 'Flow', description: 'Normalizing Flow', params: { hidden_dim: 64, num_layers: 4 } },
  ],
  'General NN': [
    { name: 'MLP', description: 'Multi-Layer Perceptron', params: { hidden_dims: [64, 32], activation: 'relu' } },
    { name: 'PINN', description: 'Physics-Informed NN', params: { hidden_dim: 64 } },
    { name: 'NODE', description: 'Neural ODE', params: { hidden_dim: 64 } },
  ],

  // Classical / Machine Learning
  'Linear Models': [
    { name: 'LinearRegression', description: 'OLS Regression', params: {} },
    { name: 'LogisticRegression', description: 'Classification', params: { C: 1.0 } },
    { name: 'Ridge', description: 'L2 Regularization', params: { alpha: 1.0 } },
    { name: 'Lasso', description: 'L1 Regularization', params: { alpha: 1.0 } },
    { name: 'ElasticNet', description: 'L1 + L2', params: { alpha: 1.0, l1_ratio: 0.5 } },
  ],
  'Decision Trees': [
    { name: 'DecisionTree', description: 'CART', params: { max_depth: 5 } },
    { name: 'RandomForest', description: 'Ensemble', params: { n_estimators: 100, max_depth: 10 } },
  ],
  'Boosting Methods': [
    { name: 'XGBoost', description: 'Extreme Gradient Boosting', params: { n_estimators: 100, learning_rate: 0.1 } },
    { name: 'LightGBM', description: 'Light Gradient Boosting', params: { n_estimators: 100, num_leaves: 31 } },
    { name: 'AdaBoost', description: 'Adaptive Boosting', params: { n_estimators: 50 } },
  ],
  'Support Vector Machines': [
    { name: 'SVM', description: 'Support Vector Machine', params: { C: 1.0, kernel: 'rbf' } },
    { name: 'SVR', description: 'Support Vector Regression', params: { C: 1.0, kernel: 'rbf' } },
  ],
  'Nearest Neighbors': [
    { name: 'kNN', description: 'k-Nearest Neighbors', params: { n_neighbors: 5 } },
  ],
  'Naive Bayes': [
    { name: 'GaussianNB', description: 'Gaussian Naive Bayes', params: {} },
  ],

  // Split Helpers
  'Clustering': [
    { name: 'KMeans', description: 'K-Means Clustering', params: { n_clusters: 8 } },
    { name: 'DBSCAN', description: 'Density-Based Clustering', params: { eps: 0.5, min_samples: 5 } },
    { name: 'Agglomerative', description: 'Hierarchical Clustering', params: { n_clusters: 2 } },
    { name: 'Spectral', description: 'Spectral Clustering', params: { n_clusters: 8 } },
    { name: 'GMM', description: 'Gaussian Mixture', params: { n_components: 1 } },
  ],
  'Dimensionality Reduction': [
    { name: 'PCA', description: 'Principal Component Analysis', params: { n_components: 10 } },
    { name: 'UMAP', description: 'Uniform Manifold Approx', params: { n_components: 2 } },
    { name: 't-SNE', description: 't-Distributed SNE', params: { n_components: 2, perplexity: 30 } },
    { name: 'ICA', description: 'Independent Component Analysis', params: { n_components: 10 } },
    { name: 'LDA', description: 'Linear Discriminant Analysis', params: { n_components: 2 } },
  ],
  'Association Rules': [
    { name: 'Apriori', description: 'Frequent Itemset Mining', params: { min_support: 0.5, min_confidence: 0.5 } },
    { name: 'Eclat', description: 'Vertical Itemset Mining', params: { min_support: 0.5 } },
    { name: 'FPGrowth', description: 'Frequent Pattern Growth', params: { min_support: 0.5, min_confidence: 0.5 } },
  ],
  'Preprocessing': [
     { name: 'StandardScaler', description: 'Normalization', params: {} },
     { name: 'MinMaxScaler', description: 'Scaling', params: {} },
  ],
};

const CATEGORY_TYPES: Record<string, 'deep' | 'mac' | 'helper'> = {
  'Recurrent Networks': 'deep',
  'Transformers': 'deep',
  'Convolutional': 'deep',
  'Autoencoders': 'deep',
  'Spiking': 'deep',
  'Memory Networks': 'deep',
  'Probabilistic': 'deep',
  'General NN': 'deep',
  
  'Linear Models': 'mac',
  'Decision Trees': 'mac',
  'Boosting Methods': 'mac',
  'Support Vector Machines': 'mac',
  'Nearest Neighbors': 'mac',
  'Naive Bayes': 'mac',
  
  'Clustering': 'helper',
  'Dimensionality Reduction': 'helper',
  'Association Rules': 'helper',
  'Preprocessing': 'helper',
};

export default function NodePalette() {
  const [searchQuery, setSearchQuery] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['Recurrent Networks', 'Clustering', 'Dimensionality Reduction']));

  const toggle = (cat: string) => {
    const next = new Set(expanded);
    if (next.has(cat)) next.delete(cat);
    else next.add(cat);
    setExpanded(next);
  };

  const filtered = Object.entries(MODEL_CATALOG).reduce((acc, [cat, models]) => {
    const f = models.filter(m => 
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.description.toLowerCase().includes(searchQuery.toLowerCase())
    );
    if (f.length) acc[cat] = f;
    return acc;
  }, {} as typeof MODEL_CATALOG);

  const handleDragStart = (e: React.DragEvent, model: { name: string; category: string; modelType: string; params: Record<string, unknown> }) => {
    e.dataTransfer.setData('application/reactflow', JSON.stringify(model));
    e.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Model Palette</h2>
        <input
          type="text"
          placeholder="Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
      </div>

      {/* Categories */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {Object.entries(filtered).map(([cat, models]) => (
          <div key={cat} className="border-b border-slate-800/50">
            <button
              onClick={() => toggle(cat)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-300 hover:bg-slate-800/50 transition-colors"
            >
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${
                  CATEGORY_TYPES[cat] === 'deep' ? 'bg-indigo-500' :
                  CATEGORY_TYPES[cat] === 'mac' ? 'bg-emerald-500' : 'bg-amber-500'
                }`} />
                <span>{cat}</span>
              </div>
              <svg 
                className={`w-4 h-4 text-slate-500 transition-transform ${expanded.has(cat) ? 'rotate-180' : ''}`} 
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {expanded.has(cat) && (
              <div className="pb-2 px-2">
                {models.map((model) => (
                  <div
                    key={model.name}
                    draggable
                    onDragStart={(e) => handleDragStart(e, {
                      name: model.name,
                      category: cat,
                      modelType: CATEGORY_TYPES[cat],
                      params: model.params,
                    })}
                    className="flex items-center gap-3 px-3 py-2 rounded-lg cursor-grab active:cursor-grabbing text-slate-400 hover:bg-slate-800 hover:text-white transition-colors group"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium truncate">{model.name}</div>
                      <div className="text-xs text-slate-500 truncate">{model.description}</div>
                    </div>
                    <svg className="w-4 h-4 text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16M4 16h16" />
                    </svg>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Data Nodes */}
      <div className="p-4 border-t border-slate-800">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Data Nodes</h3>
        <div className="space-y-2">
          <div
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/reactflow', JSON.stringify({ type: 'data', nodeKind: 'input', label: 'Data Input' }));
              e.dataTransfer.effectAllowed = 'move';
            }}
            className="flex items-center gap-2 px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg cursor-grab text-cyan-400 text-sm hover:bg-cyan-500/20 transition-colors"
          >
            <div className="w-2 h-2 rounded-full bg-cyan-500" />
            Data Input
          </div>
          <div
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData('application/reactflow', JSON.stringify({ type: 'data', nodeKind: 'output', label: 'Output' }));
              e.dataTransfer.effectAllowed = 'move';
            }}
            className="flex items-center gap-2 px-3 py-2 bg-rose-500/10 border border-rose-500/30 rounded-lg cursor-grab text-rose-400 text-sm hover:bg-rose-500/20 transition-colors"
          >
            <div className="w-2 h-2 rounded-full bg-rose-500" />
            Output
          </div>
        </div>
      </div>
    </div>
  );
}
