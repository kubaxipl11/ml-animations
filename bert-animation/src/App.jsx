import React, { useState } from 'react';
import { 
  BookOpen, Layers, Type, Brain, Target, Zap, 
  GitMerge, GraduationCap, Cpu, Settings
} from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import TokenizationPanel from './TokenizationPanel';
import EmbeddingsPanel from './EmbeddingsPanel';
import AttentionPanel from './AttentionPanel';
import EncoderLayerPanel from './EncoderLayerPanel';
import PreTrainingPanel from './PreTrainingPanel';
import FineTuningPanel from './FineTuningPanel';
import TasksPanel from './TasksPanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'overview', label: '1. Overview', icon: BookOpen, color: 'blue' },
  { id: 'tokenization', label: '2. Tokenization', icon: Type, color: 'green' },
  { id: 'embeddings', label: '3. Embeddings', icon: Layers, color: 'yellow' },
  { id: 'attention', label: '4. Self-Attention', icon: Target, color: 'purple' },
  { id: 'encoder', label: '5. Encoder Layers', icon: Cpu, color: 'pink' },
  { id: 'pretraining', label: '6. Pre-training', icon: Brain, color: 'orange' },
  { id: 'finetuning', label: '7. Fine-tuning', icon: Settings, color: 'cyan' },
  { id: 'examples', label: '8. Live Examples', icon: Zap, color: 'red' },
  { id: 'practice', label: '9. Practice Lab', icon: GraduationCap, color: 'indigo' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderPanel = () => {
    switch (activeTab) {
      case 'overview': return <OverviewPanel />;
      case 'tokenization': return <TokenizationPanel />;
      case 'embeddings': return <EmbeddingsPanel />;
      case 'attention': return <AttentionPanel />;
      case 'encoder': return <EncoderLayerPanel />;
      case 'pretraining': return <PreTrainingPanel />;
      case 'finetuning': return <FineTuningPanel />;
      case 'examples': return <TasksPanel />;
      case 'practice': return <PracticePanel />;
      default: return <OverviewPanel />;
    }
  };

  const currentIndex = tabs.findIndex(t => t.id === activeTab);

  return (
    <div className="min-h-screen text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-2xl">
            🤖
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              BERT Deep Dive
            </h1>
            <p className="text-sm text-gray-400">
              Bidirectional Encoder Representations from Transformers - Complete Guide
            </p>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="bg-black/20 px-4 py-2 overflow-x-auto">
        <div className="flex gap-1 min-w-max">
          {tabs.map((tab, index) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const isCompleted = index < currentIndex;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-all whitespace-nowrap text-sm ${
                  isActive
                    ? `bg-${tab.color}-600 text-white shadow-lg`
                    : isCompleted
                    ? `bg-${tab.color}-900/50 text-${tab.color}-300 hover:bg-${tab.color}-800/50`
                    : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-white'
                }`}
                style={isActive ? { backgroundColor: getColor(tab.color) } : {}}
              >
                <Icon size={16} />
                <span className="hidden md:inline">{tab.label}</span>
                <span className="md:hidden">{index + 1}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* Progress Bar */}
      <div className="h-1 bg-black/20 flex">
        {tabs.map((tab, index) => (
          <div
            key={tab.id}
            className={`flex-1 transition-all duration-500 ${
              index <= currentIndex
                ? 'bg-gradient-to-r from-blue-500 to-purple-500'
                : 'bg-transparent'
            }`}
          />
        ))}
      </div>

      {/* Main Content */}
      <main className="p-4 md:p-6 max-w-7xl mx-auto">
        {renderPanel()}
      </main>

      {/* Navigation Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-black/80 backdrop-blur-sm border-t border-white/10 px-6 py-3">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button
            onClick={() => currentIndex > 0 && setActiveTab(tabs[currentIndex - 1].id)}
            disabled={currentIndex === 0}
            className={`px-4 py-2 rounded-lg transition-all ${
              currentIndex === 0
                ? 'bg-white/5 text-gray-600 cursor-not-allowed'
                : 'bg-white/10 hover:bg-white/20 text-white'
            }`}
          >
            ← Previous
          </button>
          
          <div className="text-sm text-gray-400">
            Section {currentIndex + 1} of {tabs.length}
          </div>
          
          <button
            onClick={() => currentIndex < tabs.length - 1 && setActiveTab(tabs[currentIndex + 1].id)}
            disabled={currentIndex === tabs.length - 1}
            className={`px-4 py-2 rounded-lg transition-all ${
              currentIndex === tabs.length - 1
                ? 'bg-white/5 text-gray-600 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700 text-white'
            }`}
          >
            Next →
          </button>
        </div>
      </footer>

      {/* Spacer for fixed footer */}
      <div className="h-20" />
    </div>
  );
}

function getColor(color) {
  const colors = {
    blue: '#3b82f6',
    green: '#22c55e',
    yellow: '#eab308',
    purple: '#8b5cf6',
    pink: '#ec4899',
    orange: '#f97316',
    cyan: '#06b6d4',
    red: '#ef4444',
    indigo: '#6366f1',
  };
  return colors[color] || colors.blue;
}
