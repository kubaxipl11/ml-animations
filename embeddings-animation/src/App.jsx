import React, { useState } from 'react';
import AlgebraPanel from './AlgebraPanel';
import SimilarityPanel from './SimilarityPanel';
import SpacePanel from './SpacePanel';
import { Calculator, Compass, Orbit } from 'lucide-react';

const TABS = [
    { id: 'algebra', label: '1. Word Algebra', icon: Calculator },
    { id: 'similarity', label: '2. Similarity Lab', icon: Compass },
    { id: 'space', label: '3. 3D Semantic Space', icon: Orbit }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('algebra');

    const renderContent = () => {
        switch (activeTab) {
            case 'algebra': return <AlgebraPanel />;
            case 'similarity': return <SimilarityPanel />;
            case 'space': return <SpacePanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-slate-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 mb-2 tracking-tight">
                        Word Embeddings
                    </h1>
                    <p className="text-slate-400 text-lg">
                        Where words become geometry.
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
                                        ? 'bg-cyan-600 text-white shadow-lg scale-105 shadow-cyan-500/20'
                                        : 'bg-slate-800 text-slate-400 hover:bg-slate-700 shadow-sm border border-slate-700'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="bg-slate-800 rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
