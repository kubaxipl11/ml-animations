import React, { useState } from 'react';
import ResidualsPanel from './ResidualsPanel';
import InteractivePanel from './InteractivePanel';
import CostPanel from './CostPanel';
import { Ruler, MousePointer2, TrendingUp } from 'lucide-react';

const TABS = [
    { id: 'residuals', label: '1. The Residuals', icon: Ruler },
    { id: 'interactive', label: '2. Interactive Fitter', icon: MousePointer2 },
    { id: 'cost', label: '3. Cost Landscape', icon: TrendingUp }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('residuals');

    const renderContent = () => {
        switch (activeTab) {
            case 'residuals': return <ResidualsPanel />;
            case 'interactive': return <InteractivePanel />;
            case 'cost': return <CostPanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-4 font-sans text-slate-900">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-4xl font-extrabold text-slate-800 mb-2 tracking-tight">
                        Linear Regression
                    </h1>
                    <p className="text-slate-600 text-lg">
                        Finding the Line of Best Fit
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
            </div>
        </div>
    );
}
