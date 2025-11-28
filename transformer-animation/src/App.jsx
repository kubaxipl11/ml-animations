import React, { useState } from 'react';
import { Sparkles, Eye, Layers, ArrowRightLeft, Zap, GraduationCap, Building2 } from 'lucide-react';
import OverviewPanel from './OverviewPanel';
import EncoderPanel from './EncoderPanel';
import DecoderPanel from './DecoderPanel';
import DataFlowPanel from './DataFlowPanel';
import VariantsPanel from './VariantsPanel';
import PracticePanel from './PracticePanel';

const tabs = [
    { id: 'overview', label: '1. Architecture', icon: Building2, color: 'from-amber-500 to-orange-500' },
    { id: 'encoder', label: '2. Encoder', icon: Layers, color: 'from-blue-500 to-cyan-500' },
    { id: 'decoder', label: '3. Decoder', icon: Layers, color: 'from-purple-500 to-pink-500' },
    { id: 'dataflow', label: '4. Data Flow', icon: ArrowRightLeft, color: 'from-green-500 to-emerald-500' },
    { id: 'variants', label: '5. Variants', icon: Zap, color: 'from-indigo-500 to-violet-500' },
    { id: 'practice', label: '6. Practice Lab', icon: GraduationCap, color: 'from-rose-500 to-red-500' },
];

export default function App() {
    const [activeTab, setActiveTab] = useState('overview');

    const renderPanel = () => {
        switch (activeTab) {
            case 'overview':
                return <OverviewPanel />;
            case 'encoder':
                return <EncoderPanel />;
            case 'decoder':
                return <DecoderPanel />;
            case 'dataflow':
                return <DataFlowPanel />;
            case 'variants':
                return <VariantsPanel />;
            case 'practice':
                return <PracticePanel />;
            default:
                return <OverviewPanel />;
        }
    };

    return (
        <div className="min-h-screen">
            {/* Header */}
            <header className="bg-slate-900/80 backdrop-blur-sm border-b border-slate-700 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 py-4">
                    <div className="flex items-center gap-3">
                        <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-xl">
                            <Sparkles className="text-white" size={28} />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold text-white">
                                Transformer Architecture
                            </h1>
                            <p className="text-slate-400 text-sm">
                                "Attention Is All You Need" - The Complete Picture
                            </p>
                        </div>
                    </div>
                </div>
            </header>

            {/* Tab Navigation */}
            <nav className="bg-slate-800/50 border-b border-slate-700 sticky top-[73px] z-40">
                <div className="max-w-7xl mx-auto px-4">
                    <div className="flex gap-1 overflow-x-auto py-2">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-all ${
                                    activeTab === tab.id
                                        ? `bg-gradient-to-r ${tab.color} text-white shadow-lg`
                                        : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                                }`}
                            >
                                <tab.icon size={18} />
                                {tab.label}
                            </button>
                        ))}
                    </div>
                    {/* Progress indicator */}
                    <div className="flex gap-1 pb-2">
                        {tabs.map((tab, i) => (
                            <div 
                                key={tab.id}
                                className={`h-1 flex-1 rounded-full transition-all ${
                                    tabs.findIndex(t => t.id === activeTab) >= i
                                        ? `bg-gradient-to-r ${tab.color}`
                                        : 'bg-slate-700'
                                }`}
                            />
                        ))}
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto">
                {renderPanel()}
            </main>

            {/* Footer */}
            <footer className="border-t border-slate-700 mt-8 py-4">
                <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
                    💡 Tip: Progress through each tab in order for the best learning experience
                </div>
            </footer>
        </div>
    );
}
