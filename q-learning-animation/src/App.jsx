import React, { useState } from 'react';
import TablePanel from './TablePanel';
import AlgorithmPanel from './AlgorithmPanel';
import TrainingPanel from './TrainingPanel';
import { Grid, Calculator, PlayCircle } from 'lucide-react';

const TABS = [
    { id: 'table', label: '1. The Q-Table (Brain)', icon: Grid },
    { id: 'algorithm', label: '2. The Bellman Update', icon: Calculator },
    { id: 'training', label: '3. Training Loop', icon: PlayCircle }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('table');

    const renderContent = () => {
        switch (activeTab) {
            case 'table': return <TablePanel />;
            case 'algorithm': return <AlgorithmPanel />;
            case 'training': return <TrainingPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-cyan-950 to-blue-950 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-400 to-blue-400 mb-2 tracking-tight">
                        Q-Learning
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Part 2: How the Agent Learns.
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
                                        ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg scale-105'
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
