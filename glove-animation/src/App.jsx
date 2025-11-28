import React, { useState } from 'react';
import { BookOpen, Grid3X3, Calculator, GitCompare, Code, GraduationCap, ChevronLeft, ChevronRight, Sparkles } from 'lucide-react';
import IntroPanel from './IntroPanel';
import CooccurrencePanel from './CooccurrencePanel';
import ObjectivePanel from './ObjectivePanel';
import ComparisonPanel from './ComparisonPanel';
import CodePanel from './CodePanel';
import PracticePanel from './PracticePanel';

const tabs = [
  { id: 'intro', label: '1. Introduction', icon: BookOpen },
  { id: 'cooccurrence', label: '2. Co-occurrence', icon: Grid3X3 },
  { id: 'objective', label: '3. GloVe Objective', icon: Calculator },
  { id: 'comparison', label: '4. vs Word2Vec', icon: GitCompare },
  { id: 'code', label: '5. Python Code', icon: Code },
  { id: 'practice', label: '6. Practice Lab', icon: GraduationCap }
];

export default function App() {
  const [activeTab, setActiveTab] = useState('intro');

  const currentIndex = tabs.findIndex(t => t.id === activeTab);
  
  const goToPrevious = () => {
    if (currentIndex > 0) {
      setActiveTab(tabs[currentIndex - 1].id);
    }
  };

  const goToNext = () => {
    if (currentIndex < tabs.length - 1) {
      setActiveTab(tabs[currentIndex + 1].id);
    }
  };

  const renderPanel = () => {
    switch (activeTab) {
      case 'intro': return <IntroPanel />;
      case 'cooccurrence': return <CooccurrencePanel />;
      case 'objective': return <ObjectivePanel />;
      case 'comparison': return <ComparisonPanel />;
      case 'code': return <CodePanel />;
      case 'practice': return <PracticePanel />;
      default: return <IntroPanel />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-black/30 backdrop-blur-sm border-b border-white/10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🔮</span>
            <div>
              <h1 className="text-xl font-bold gradient-text">GloVe</h1>
              <p className="text-xs text-gray-400">Global Vectors for Word Representation</p>
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="bg-black/20 border-b border-white/10 px-4 py-2 overflow-x-auto">
        <div className="max-w-7xl mx-auto flex gap-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all ${
                  activeTab === tab.id
                    ? 'bg-violet-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-white/10'
                }`}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-6xl mx-auto">
          {renderPanel()}
        </div>
      </main>

      {/* Footer Navigation */}
      <footer className="bg-black/30 backdrop-blur-sm border-t border-white/10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <button
            onClick={goToPrevious}
            disabled={currentIndex === 0}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              currentIndex === 0
                ? 'text-gray-600 cursor-not-allowed'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <ChevronLeft size={18} />
            Previous
          </button>
          
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span>Section</span>
            <span className="text-violet-400 font-bold">{currentIndex + 1}</span>
            <span>of</span>
            <span className="text-violet-400 font-bold">{tabs.length}</span>
          </div>

          <button
            onClick={goToNext}
            disabled={currentIndex === tabs.length - 1}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              currentIndex === tabs.length - 1
                ? 'text-gray-600 cursor-not-allowed'
                : 'bg-violet-600 text-white hover:bg-violet-700'
            }`}
          >
            Next
            <ChevronRight size={18} />
          </button>
        </div>
      </footer>
    </div>
  );
}
