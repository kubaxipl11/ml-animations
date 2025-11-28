import React, { useState } from 'react';
import ConceptPanel from './ConceptPanel';
import BPEPanel from './BPEPanel';
import WordPiecePanel from './WordPiecePanel';
import PracticePanel from './PracticePanel';
import { BookOpen, Scissors, Puzzle, FlaskConical } from 'lucide-react';

const TABS = [
    { id: 'concept', label: '1. What is Tokenization?', icon: BookOpen },
    { id: 'bpe', label: '2. BPE Algorithm', icon: Scissors },
    { id: 'wordpiece', label: '3. WordPiece', icon: Puzzle },
    { id: 'practice', label: '4. Practice Lab', icon: FlaskConical }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('concept');

    const renderContent = () => {
        switch (activeTab) {
            case 'concept': return <ConceptPanel />;
            case 'bpe': return <BPEPanel />;
            case 'wordpiece': return <WordPiecePanel />;
            case 'practice': return <PracticePanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-4 font-sans text-slate-900">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-4xl font-extrabold text-slate-800 mb-2 tracking-tight">
                        Tokenization Deep Dive
                    </h1>
                    <p className="text-slate-600 text-lg">
                        How LLMs break text into learnable pieces
                    </p>
                </header>

                {/* Navigation Tabs */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                    {TABS.map(tab => {
                        const Icon = tab.icon;
                        return (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold transition-all transform hover:scale-105 ${activeTab === tab.id
                                        ? 'bg-indigo-600 text-white shadow-lg scale-105'
                                        : 'bg-white text-slate-600 hover:bg-slate-100 shadow-sm border border-slate-200'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="bg-white rounded-2xl shadow-xl border border-slate-200 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>

                <footer className="mt-8 text-center text-slate-500 text-sm">
                    <p>Interactive tokenization learning - adjust parameters and see real-time results</p>
                </footer>
            </div>
        </div>
    );
}
