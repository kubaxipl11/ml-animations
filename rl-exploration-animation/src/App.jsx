import React, { useState } from 'react';
import EpsilonPanel from './EpsilonPanel';
import CliffPanel from './CliffPanel';
import HyperparameterPanel from './HyperparameterPanel';
import { Compass, AlertTriangle, Sliders } from 'lucide-react';

const TABS = [
    { id: 'epsilon', label: '1. Epsilon-Greedy (Exploration)', icon: Compass },
    { id: 'cliff', label: '2. The Cliff (Risk vs Reward)', icon: AlertTriangle },
    { id: 'hyperparams', label: '3. Hyperparameter Tuning', icon: Sliders }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('epsilon');

    const renderContent = () => {
        switch (activeTab) {
            case 'epsilon': return <EpsilonPanel />;
            case 'cliff': return <CliffPanel />;
            case 'hyperparams': return <HyperparameterPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-violet-950 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-violet-400 to-fuchsia-400 mb-2 tracking-tight">
                        Exploration & Optimization
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Part 3: Mastering the Trade-offs.
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
                                        ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-lg scale-105'
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
