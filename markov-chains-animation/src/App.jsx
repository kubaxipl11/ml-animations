import React, { useState } from 'react';
import PropertyPanel from './PropertyPanel';
import BuilderPanel from './BuilderPanel';
import StationaryPanel from './StationaryPanel';
import TextPanel from './TextPanel';
import { Footprints, Network, Scale, FileText } from 'lucide-react';

const TABS = [
    { id: 'property', label: '1. The Markov Property', icon: Footprints },
    { id: 'builder', label: '2. Transition Matrix', icon: Network },
    { id: 'stationary', label: '3. Stationary Distribution', icon: Scale },
    { id: 'text', label: '4. Text Generation', icon: FileText }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('property');

    const renderContent = () => {
        switch (activeTab) {
            case 'property': return <PropertyPanel />;
            case 'builder': return <BuilderPanel />;
            case 'stationary': return <StationaryPanel />;
            case 'text': return <TextPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-indigo-950 p-4 font-sans text-slate-100">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-indigo-400 to-violet-400 mb-2 tracking-tight">
                        Markov Chains
                    </h1>
                    <p className="text-slate-300 text-lg">
                        From random walks to AI text generation.
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
                                        ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-105'
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
