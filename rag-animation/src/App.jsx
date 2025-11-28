import React, { useState } from 'react';
import ConceptPanel from './ConceptPanel';
import PipelinePanel from './PipelinePanel';
import RetrievalPanel from './RetrievalPanel';
import PracticePanel from './PracticePanel';
import { BookOpen, GitBranch, Search, FlaskConical } from 'lucide-react';

const TABS = [
    { id: 'concept', label: '1. What is RAG?', icon: BookOpen },
    { id: 'pipeline', label: '2. RAG Pipeline', icon: GitBranch },
    { id: 'retrieval', label: '3. Vector Search', icon: Search },
    { id: 'practice', label: '4. Practice Lab', icon: FlaskConical }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('concept');

    const renderContent = () => {
        switch (activeTab) {
            case 'concept': return <ConceptPanel />;
            case 'pipeline': return <PipelinePanel />;
            case 'retrieval': return <RetrievalPanel />;
            case 'practice': return <PracticePanel />;
            default: return null;
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-4 font-sans text-slate-900">
            <div className="max-w-7xl mx-auto">
                <header className="mb-6 text-center">
                    <h1 className="text-4xl font-extrabold text-slate-800 mb-2 tracking-tight">
                        Retrieval-Augmented Generation
                    </h1>
                    <p className="text-slate-600 text-lg">
                        Give LLMs access to external knowledge
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

                <footer className="mt-8 text-center text-slate-500 text-sm">
                    <p>Learn how RAG enhances LLMs with external knowledge retrieval</p>
                </footer>
            </div>
        </div>
    );
}
