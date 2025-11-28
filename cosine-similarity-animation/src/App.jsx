import React, { useState } from 'react';
import DotProductPanel from './DotProductPanel';
import RecommenderPanel from './RecommenderPanel';
import SearchPanel from './SearchPanel';
import { Calculator, Film, Search } from 'lucide-react';

const TABS = [
    { id: 'dot', label: '1. The Dot Product', icon: Calculator },
    { id: 'recommender', label: '2. Movie Matcher', icon: Film },
    { id: 'search', label: '3. Search Engine', icon: Search }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('dot');

    const renderContent = () => {
        switch (activeTab) {
            case 'dot': return <DotProductPanel />;
            case 'recommender': return <RecommenderPanel />;
            case 'search': return <SearchPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-purple-900 to-pink-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 mb-2 tracking-tight">
                        Cosine Similarity
                    </h1>
                    <p className="text-slate-300 text-lg">
                        The math behind recommendations and search.
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
                                        ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg scale-105'
                                        : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/50 shadow-sm border border-slate-700'
                                    }`}
                            >
                                <Icon size={20} />
                                {tab.label}
                            </button>
                        );
                    })}
                </div>

                {/* Main Content Area */}
                <div className="bg-slate-900/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-slate-700 overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>
            </div>
        </div>
    );
}
