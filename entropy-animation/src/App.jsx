import React, { useState } from 'react';
import SurprisePanel from './SurprisePanel';
import EntropyPanel from './EntropyPanel';
import { Lightbulb, BarChart3 } from 'lucide-react';

const TABS = [
    { id: 'surprise', label: '1. The Bit (Surprise)', icon: Lightbulb },
    { id: 'entropy', label: '2. Entropy (Uncertainty)', icon: BarChart3 }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('surprise');

    const renderContent = () => {
        switch (activeTab) {
            case 'surprise': return <SurprisePanel />;
            case 'entropy': return <EntropyPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-pink-950 to-rose-950 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 via-rose-400 to-red-400 mb-2 tracking-tight">
                        Information Theory
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Part 1: Bits, Surprise, and Entropy.
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
                                        ? 'bg-gradient-to-r from-pink-600 to-rose-600 text-white shadow-lg scale-105'
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
