import React, { useState } from 'react';
import TutorialModePanel from './TutorialModePanel';
import GeometricVisualizerPanel from './GeometricVisualizerPanel';
import InteractiveExplorerPanel from './InteractiveExplorerPanel';
import AnimationPanel from './AnimationPanel';
import PracticePanel from './PracticePanel';

const TABS = [
    { id: 'tutorial', label: '📚 Tutorial', icon: '📚' },
    { id: 'geometric', label: '🌐 Geometric View', icon: '🌐' },
    { id: 'interactive', label: '🎮 Interactive', icon: '🎮' },
    { id: 'animation', label: '🎬 Decomposition', icon: '🎬' },
    { id: 'practice', label: '✏️ Practice', icon: '✏️' }
];

export default function App() {
    const [activeTab, setActiveTab] = useState('tutorial');

    const renderContent = () => {
        switch (activeTab) {
            case 'tutorial':
                return <TutorialModePanel />;
            case 'geometric':
                return <GeometricVisualizerPanel />;
            case 'interactive':
                return <InteractiveExplorerPanel />;
            case 'animation':
                return (
                    <div className="flex flex-col lg:flex-row gap-4 h-full">
                        <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
                            <AnimationPanel />
                        </div>
                    </div>
                );
            case 'practice':
                return (
                    <div className="flex flex-col lg:flex-row gap-4 h-full">
                        <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
                            <PracticePanel />
                        </div>
                    </div>
                );
            default:
                return null;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-100 to-gray-200 p-4">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-4xl font-bold text-gray-800 text-center mb-2">
                    Eigenvalue Decomposition
                </h1>
                <p className="text-center text-gray-600 mb-6">
                    Understanding eigenvalues from first principles
                </p>

                {/* Tab Navigation */}
                <div className="flex flex-wrap justify-center gap-2 mb-6">
                    {TABS.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`px-6 py-3 font-bold rounded-lg transition-all transform hover:scale-105 ${activeTab === tab.id
                                    ? 'bg-blue-600 text-white shadow-lg scale-105'
                                    : 'bg-white text-gray-700 hover:bg-gray-100 shadow'
                                }`}
                        >
                            <span className="text-xl mr-2">{tab.icon}</span>
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Content Area */}
                <div className="bg-gray-50 rounded-2xl shadow-2xl overflow-hidden min-h-[600px]">
                    {renderContent()}
                </div>

                {/* Info Footer */}
                <div className="mt-4 p-4 bg-white rounded-lg shadow text-center">
                    <p className="text-sm text-gray-600">
                        <strong>💡 Learning Path:</strong> Start with Tutorial → Explore Geometric View →
                        Try Interactive Mode → Watch Decomposition → Practice Exercises
                    </p>
                </div>
            </div>
        </div>
    );
}
