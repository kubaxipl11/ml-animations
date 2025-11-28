import React, { useState } from 'react';
import ConditionalPanel from './ConditionalPanel';
import BayesPanel from './BayesPanel';
import MedicalPanel from './MedicalPanel';
import { Dices, RefreshCw, Stethoscope } from 'lucide-react';

const TABS = [
    { id: 'conditional', label: '1. Conditional Probability', icon: Dices },
    { id: 'bayes', label: '2. Bayes\' Theorem', icon: RefreshCw },
    { id: 'medical', label: '3. Medical Testing', icon: Stethoscope }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('conditional');

    const renderContent = () => {
        switch (activeTab) {
            case 'conditional': return <ConditionalPanel />;
            case 'bayes': return <BayesPanel />;
            case 'medical': return <MedicalPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-emerald-950 via-teal-900 to-cyan-900 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 mb-2 tracking-tight">
                        Conditional Probability & Bayes' Theorem
                    </h1>
                    <p className="text-slate-300 text-lg">
                        Updating beliefs with evidence.
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
                                        ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-white shadow-lg scale-105'
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
