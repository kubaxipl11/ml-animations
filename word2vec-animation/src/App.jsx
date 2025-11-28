import React, { useState } from 'react';
import { BookOpen, Target, Shuffle, Zap, Brain, Code, GraduationCap, ArrowLeft, ArrowRight } from 'lucide-react';
import IntroPanel from './IntroPanel';
import SkipGramPanel from './SkipGramPanel';
import CbowPanel from './CbowPanel';
import NegativeSamplingPanel from './NegativeSamplingPanel';
import EmbeddingsPanel from './EmbeddingsPanel';
import CodePanel from './CodePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'intro', label: '1. Introduction', icon: BookOpen, color: 'blue' },
  { id: 'skipgram', label: '2. Skip-gram', icon: Target, color: 'green' },
  { id: 'cbow', label: '3. CBOW', icon: Shuffle, color: 'yellow' },
  { id: 'negative', label: '4. Negative Sampling', icon: Zap, color: 'orange' },
  { id: 'embeddings', label: '5. Embeddings', icon: Brain, color: 'purple' },
  { id: 'code', label: '6. Python Code', icon: Code, color: 'cyan' },
  { id: 'practice', label: '7. Practice Lab', icon: GraduationCap, color: 'pink' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('intro');

  const renderPanel = () => {
    switch (activeTab) {
      case 'intro': return <IntroPanel />;
      case 'skipgram': return <SkipGramPanel />;
      case 'cbow': return <CbowPanel />;
      case 'negative': return <NegativeSamplingPanel />;
      case 'embeddings': return <EmbeddingsPanel />;
      case 'code': return <CodePanel />;
      case 'practice': return <PracticePanel />;
      default: return <IntroPanel />;
    }
  };

  const currentIndex = tabs.findIndex(t => t.id === activeTab);

  return (
    <div className="min-h-screen text-white">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🎯</span>
            <div>
              <h1 className="text-xl font-bold text-white">Word2Vec</h1>
              <p className="text-xs text-gray-400">Words as Vectors in Continuous Space</p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-black/20 border-b border-white/10 overflow-x-auto">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 py-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                    isActive
                      ? `bg-${tab.color}-600 text-white shadow-lg`
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                  style={isActive ? { backgroundColor: getColor(tab.color) } : {}}
                >
                  <Icon size={16} />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {renderPanel()}
      </main>

      {/* Footer Navigation */}
      <footer className="fixed bottom-0 left-0 right-0 bg-black/50 backdrop-blur-sm border-t border-white/10 p-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button
            onClick={() => currentIndex > 0 && setActiveTab(tabs[currentIndex - 1].id)}
            disabled={currentIndex === 0}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              currentIndex === 0
                ? 'text-gray-600 cursor-not-allowed'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <ArrowLeft size={18} />
            Previous
          </button>
          <span className="text-gray-500 text-sm">
            Section <span className="text-white font-medium">{currentIndex + 1}</span> of {tabs.length}
          </span>
          <button
            onClick={() => currentIndex < tabs.length - 1 && setActiveTab(tabs[currentIndex + 1].id)}
            disabled={currentIndex === tabs.length - 1}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              currentIndex === tabs.length - 1
                ? 'text-gray-600 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            Next
            <ArrowRight size={18} />
          </button>
        </div>
      </footer>
    </div>
  );
}

function getColor(colorName) {
  const colors = {
    blue: '#2563eb',
    green: '#16a34a',
    yellow: '#ca8a04',
    orange: '#ea580c',
    purple: '#9333ea',
    cyan: '#0891b2',
    pink: '#db2777',
    red: '#dc2626',
  };
  return colors[colorName] || colors.blue;
}
