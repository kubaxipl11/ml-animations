import React, { useState } from 'react';
import AnimationPanel from './AnimationPanel';
import PracticePanel from './PracticePanel';
import LeakyReluGraphPanel from './LeakyReluGraphPanel';

export default function App() {
  const [animationStep, setAnimationStep] = useState(0);
  const [zValue, setZValue] = useState(null);
  const [leakyReluValue, setLeakyReluValue] = useState(null);

  const [practiceZ, setPracticeZ] = useState(null);
  const [practiceLeakyRelu, setPracticeLeakyRelu] = useState(null);

  const handleAnimationStepChange = (step, z, leakyRelu) => {
    setAnimationStep(step);
    if (step >= 5) {
      setZValue(z);
    } else {
      setZValue(null);
    }
    if (step >= 6) {
      setLeakyReluValue(leakyRelu);
    } else {
      setLeakyReluValue(null);
    }
  };

  const handlePracticeStepChange = (step, z, leakyRelu) => {
    if (step >= 2) {
      setPracticeZ(z);
    } else {
      setPracticeZ(null);
    }
    if (step >= 3) {
      setPracticeLeakyRelu(leakyRelu);
    } else {
      setPracticeLeakyRelu(null);
    }
  };

  // Determine which values to show on the graph (animation takes priority if active)
  const graphZ = zValue !== null ? zValue : practiceZ;
  const graphLeakyRelu = leakyReluValue !== null ? leakyReluValue : practiceLeakyRelu;
  const isGraphActive = graphZ !== null;

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold text-gray-800 text-center mb-4">Leaky ReLU Activation Function</h1>

      <div className="flex flex-col gap-4 max-w-7xl mx-auto">
        {/* Top Row - Animation and Practice */}
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Left Panel - Animation Demo */}
          <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
            <AnimationPanel onStepChange={handleAnimationStepChange} />
          </div>

          {/* Right Panel - Interactive Practice */}
          <div className="flex-1 bg-gray-50 rounded-xl shadow-lg overflow-hidden">
            <PracticePanel onStepChange={handlePracticeStepChange} />
          </div>
        </div>

        {/* Bottom Panel - Leaky ReLU Graph */}
        <div className="bg-gray-50 rounded-xl shadow-lg overflow-hidden p-4">
          <h2 className="text-xl font-bold text-gray-800 text-center mb-3">Leaky ReLU Function Visualization</h2>
          <div className="flex justify-center">
            <LeakyReluGraphPanel
              zValue={graphZ}
              leakyReluValue={graphLeakyRelu}
              isActive={isGraphActive}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
