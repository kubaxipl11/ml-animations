import React, { useState } from 'react';
import GradientDescentPanel from './GradientDescentPanel';
import LossHistoryPanel from './LossHistoryPanel';
import PracticePanel from './PracticePanel';

export default function App() {
    const [learningRate, setLearningRate] = useState(0.1);
    const [startWeight, setStartWeight] = useState(3.0);
    const [history, setHistory] = useState([]);

    const handleParamsChange = (lr, sw) => {
        setLearningRate(lr);
        setStartWeight(sw);
        // Clear history when params change
        setHistory([]);
    };

    const handleStepChange = (iteration, weight, loss) => {
        setHistory(prev => {
            const newHistory = [...prev];
            newHistory[iteration] = { iteration, weight, loss };
            return newHistory;
        });
    };

    return (
        <div className="min-h-screen bg-gray-100 p-4">
            <h1 className="text-3xl font-bold text-gray-800 text-center mb-4">Gradient Descent</h1>
            <p className="text-center text-gray-600 mb-4">
                Visualizing how learning rate affects optimization
            </p>

            <div className="flex flex-col gap-4 max-w-7xl mx-auto">
                {/* Top Row - Animation and Controls */}
                <div className="flex flex-col lg:flex-row gap-4">
                    {/* Left Panel - Animation */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
                        <GradientDescentPanel
                            learningRate={learningRate}
                            startWeight={startWeight}
                            onStepChange={handleStepChange}
                        />
                    </div>

                    {/* Right Panel - Controls */}
                    <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
                        <PracticePanel onParamsChange={handleParamsChange} />
                    </div>
                </div>

                {/* Bottom Panel - Loss History */}
                <div className="bg-gray-50 rounded-xl shadow-lg overflow-hidden p-4">
                    <h2 className="text-xl font-bold text-gray-800 text-center mb-3">Loss History</h2>
                    <div className="flex justify-center">
                        <LossHistoryPanel history={history} />
                    </div>
                </div>
            </div>
        </div>
    );
}
