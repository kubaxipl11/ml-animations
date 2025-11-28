import React, { useState } from 'react';
import GraphCanvas from './GraphCanvas';
import SurferPanel from './SurferPanel';
import IterativePanel from './IterativePanel';
import { Share2, Waves, Calculator } from 'lucide-react';

const TABS = [
    { id: 'builder', label: '1. Graph Builder', icon: Share2 },
    { id: 'surfer', label: '2. Random Surfer', icon: Waves },
    { id: 'iterative', label: '3. Power Method', icon: Calculator }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('builder');
    // Shared graph state
    const [nodes, setNodes] = useState([
        { id: 'A', x: 200, y: 200 },
        { id: 'B', x: 400, y: 200 },
        { id: 'C', x: 300, y: 400 }
    ]);
    const [links, setLinks] = useState([
        { source: 'A', target: 'B' },
        { source: 'B', target: 'C' },
        { source: 'C', target: 'A' }
    ]);

    const renderContent = () => {
        switch (activeTab) {
            case 'builder':
                return <GraphCanvas nodes={nodes} setNodes={setNodes} links={links} setLinks={setLinks} />;
            case 'surfer':
                return <SurferPanel nodes={nodes} links={links} />;
            case 'iterative':
                return <IterativePanel nodes={nodes} links={links} />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-4 font-sans text-slate-900">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-4xl font-extrabold text-slate-800 mb-2 tracking-tight">
                        PageRank
                    </h1>
                    <p className="text-slate-600 text-lg">
                        The Algorithm that Built the Web
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
