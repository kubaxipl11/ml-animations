import React, { useState, useRef, useEffect } from 'react';
import gsap from 'gsap';

// Neural Network Architecture
const NETWORK = {
  layers: [
    { name: 'Input', neurons: 2, color: '#60a5fa' },
    { name: 'Hidden 1', neurons: 4, color: '#a78bfa' },
    { name: 'Hidden 2', neurons: 4, color: '#f472b6' },
    { name: 'Output', neurons: 1, color: '#34d399' },
  ],
};

// Activation functions
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const relu = (x) => Math.max(0, x);
const sigmoidDerivative = (x) => sigmoid(x) * (1 - sigmoid(x));

function App() {
  const [mode, setMode] = useState('intro'); // intro, forward, backward, training
  const [step, setStep] = useState(0);
  const [activations, setActivations] = useState([]);
  const [weights, setWeights] = useState([]);
  const [gradients, setGradients] = useState([]);
  const [input, setInput] = useState([1, 0]);
  const [target, setTarget] = useState(1);
  const [loss, setLoss] = useState(null);
  const [epoch, setEpoch] = useState(0);
  const svgRef = useRef(null);
  const neuronRefs = useRef({});
  const connectionRefs = useRef({});

  // Initialize weights
  useEffect(() => {
    initializeWeights();
  }, []);

  const initializeWeights = () => {
    const newWeights = [];
    for (let l = 0; l < NETWORK.layers.length - 1; l++) {
      const fromSize = NETWORK.layers[l].neurons;
      const toSize = NETWORK.layers[l + 1].neurons;
      const layerWeights = [];
      for (let i = 0; i < fromSize; i++) {
        const row = [];
        for (let j = 0; j < toSize; j++) {
          row.push((Math.random() - 0.5) * 2);
        }
        layerWeights.push(row);
      }
      newWeights.push(layerWeights);
    }
    setWeights(newWeights);
    setActivations([input.map(x => ({ value: x, preActivation: x }))]);
    setGradients([]);
  };

  // Calculate positions - center neurons vertically in each layer
  const getPosition = (layerIdx, neuronIdx) => {
    const width = 900;
    const height = 500;
    const layerSpacing = width / (NETWORK.layers.length + 1);
    const layer = NETWORK.layers[layerIdx];
    const numNeurons = layer.neurons;
    
    // Calculate vertical spacing to center the neurons
    const neuronGap = 80; // Fixed gap between neurons
    const totalHeight = (numNeurons - 1) * neuronGap;
    const startY = (height - totalHeight) / 2;
    
    return {
      x: (layerIdx + 1) * layerSpacing,
      y: startY + neuronIdx * neuronGap,
    };
  };

  // Forward propagation step
  const forwardStep = () => {
    if (step >= NETWORK.layers.length - 1) {
      const output = activations[activations.length - 1][0].value;
      const mse = Math.pow(output - target, 2) / 2;
      setLoss(mse);
      setMode('forward-complete');
      return;
    }

    const currentActivations = [...activations];
    const layerActivations = currentActivations[step];
    const nextLayerSize = NETWORK.layers[step + 1].neurons;
    const layerWeights = weights[step];
    
    const nextActivations = [];
    for (let j = 0; j < nextLayerSize; j++) {
      let sum = 0;
      for (let i = 0; i < layerActivations.length; i++) {
        sum += layerActivations[i].value * layerWeights[i][j];
      }
      // Add bias (simplified)
      sum += 0.1;
      
      const activated = step < NETWORK.layers.length - 2 ? relu(sum) : sigmoid(sum);
      nextActivations.push({ value: activated, preActivation: sum });
    }
    
    currentActivations.push(nextActivations);
    setActivations(currentActivations);
    
    // Animate
    animateForwardPropagation(step);
    setStep(step + 1);
  };

  // Backward propagation step
  const backwardStep = () => {
    const backStep = NETWORK.layers.length - 2 - (step - (NETWORK.layers.length - 1));
    
    if (backStep < 0) {
      setMode('backward-complete');
      return;
    }

    const currentGradients = [...gradients];
    
    if (backStep === NETWORK.layers.length - 2) {
      // Output layer gradient
      const output = activations[activations.length - 1][0];
      const error = output.value - target;
      const delta = error * sigmoidDerivative(output.preActivation);
      currentGradients.push([delta]);
    } else {
      // Hidden layer gradients
      const nextGradients = currentGradients[currentGradients.length - 1];
      const layerWeights = weights[backStep + 1];
      const layerActivations = activations[backStep + 1];
      
      const layerGradients = [];
      for (let i = 0; i < layerActivations.length; i++) {
        let sum = 0;
        for (let j = 0; j < nextGradients.length; j++) {
          sum += nextGradients[j] * layerWeights[i][j];
        }
        // ReLU derivative
        const grad = layerActivations[i].preActivation > 0 ? sum : 0;
        layerGradients.push(grad);
      }
      currentGradients.push(layerGradients);
    }
    
    setGradients(currentGradients);
    animateBackwardPropagation(backStep);
    setStep(step + 1);
  };

  // Animation for forward propagation
  const animateForwardPropagation = (layerIdx) => {
    const layer = NETWORK.layers[layerIdx];
    const nextLayer = NETWORK.layers[layerIdx + 1];
    
    // Animate connections
    for (let i = 0; i < layer.neurons; i++) {
      for (let j = 0; j < nextLayer.neurons; j++) {
        const key = `${layerIdx}-${i}-${j}`;
        const line = connectionRefs.current[key];
        if (line) {
          gsap.fromTo(line,
            { strokeDashoffset: 100, stroke: '#4ade80' },
            { 
              strokeDashoffset: 0, 
              duration: 0.5, 
              delay: i * 0.1,
              ease: 'power2.out',
              onComplete: () => {
                gsap.to(line, { stroke: '#6366f1', duration: 0.3 });
              }
            }
          );
        }
      }
    }
    
    // Animate neurons lighting up
    for (let j = 0; j < nextLayer.neurons; j++) {
      const key = `${layerIdx + 1}-${j}`;
      const circle = neuronRefs.current[key];
      if (circle) {
        gsap.to(circle, {
          scale: 1.3,
          duration: 0.2,
          delay: 0.5 + j * 0.1,
          yoyo: true,
          repeat: 1,
          ease: 'power2.out',
        });
      }
    }
  };

  // Animation for backward propagation
  const animateBackwardPropagation = (layerIdx) => {
    const layer = NETWORK.layers[layerIdx];
    const nextLayer = NETWORK.layers[layerIdx + 1];
    
    for (let i = 0; i < layer.neurons; i++) {
      for (let j = 0; j < nextLayer.neurons; j++) {
        const key = `${layerIdx}-${i}-${j}`;
        const line = connectionRefs.current[key];
        if (line) {
          gsap.fromTo(line,
            { stroke: '#ef4444' },
            { 
              stroke: '#f97316', 
              duration: 0.5, 
              delay: j * 0.1,
              yoyo: true,
              repeat: 1,
              ease: 'power2.inOut'
            }
          );
        }
      }
    }
  };

  // Update weights
  const updateWeights = () => {
    const learningRate = 0.1;
    const newWeights = [...weights];
    
    for (let l = 0; l < weights.length; l++) {
      const gradientLayer = gradients[weights.length - 1 - l];
      if (!gradientLayer) continue;
      
      for (let i = 0; i < weights[l].length; i++) {
        for (let j = 0; j < weights[l][i].length; j++) {
          const activation = activations[l][i].value;
          const gradient = gradientLayer[j] || 0;
          newWeights[l][i][j] -= learningRate * gradient * activation;
        }
      }
    }
    
    setWeights(newWeights);
    setEpoch(epoch + 1);
  };

  // Reset for new iteration
  const resetForNewIteration = () => {
    setActivations([input.map(x => ({ value: x, preActivation: x }))]);
    setGradients([]);
    setStep(0);
    setLoss(null);
    setMode('forward');
  };

  // Render connections
  const renderConnections = () => {
    const connections = [];
    for (let l = 0; l < NETWORK.layers.length - 1; l++) {
      const fromLayer = NETWORK.layers[l];
      const toLayer = NETWORK.layers[l + 1];
      
      for (let i = 0; i < fromLayer.neurons; i++) {
        for (let j = 0; j < toLayer.neurons; j++) {
          const from = getPosition(l, i);
          const to = getPosition(l + 1, j);
          const key = `${l}-${i}-${j}`;
          const weight = weights[l] ? weights[l][i]?.[j] : 0;
          const strokeWidth = Math.abs(weight) * 2 + 0.5;
          
          connections.push(
            <line
              key={key}
              ref={(el) => (connectionRefs.current[key] = el)}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="#6366f1"
              strokeWidth={strokeWidth}
              strokeOpacity={0.6}
              strokeDasharray="100"
              strokeDashoffset="0"
            />
          );
        }
      }
    }
    return connections;
  };

  // Render neurons
  const renderNeurons = () => {
    const neurons = [];
    NETWORK.layers.forEach((layer, layerIdx) => {
      for (let i = 0; i < layer.neurons; i++) {
        const pos = getPosition(layerIdx, i);
        const key = `${layerIdx}-${i}`;
        const activation = activations[layerIdx] ? activations[layerIdx][i]?.value : 0;
        const gradient = gradients[NETWORK.layers.length - 2 - layerIdx]?.[i] || 0;
        
        neurons.push(
          <g key={key} transform={`translate(${pos.x}, ${pos.y})`}>
            {/* Glow effect */}
            <circle
              r={28}
              fill={layer.color}
              opacity={0.3}
              filter="blur(8px)"
            />
            {/* Main neuron */}
            <circle
              ref={(el) => (neuronRefs.current[key] = el)}
              r={24}
              fill={`url(#gradient-${layerIdx})`}
              stroke={layer.color}
              strokeWidth={2}
              className="cursor-pointer hover:scale-110 transition-transform"
            />
            {/* Activation value */}
            <text
              textAnchor="middle"
              dy="5"
              fill="white"
              fontSize="12"
              fontWeight="bold"
            >
              {activation?.toFixed(2) || '0.00'}
            </text>
            {/* Gradient indicator (during backprop) */}
            {mode.includes('backward') && gradient !== 0 && (
              <text
                textAnchor="middle"
                dy="35"
                fill="#f97316"
                fontSize="10"
              >
                δ={gradient.toFixed(3)}
              </text>
            )}
          </g>
        );
      }
    });
    return neurons;
  };

  // Render layer labels
  const renderLayerLabels = () => {
    return NETWORK.layers.map((layer, idx) => {
      const pos = getPosition(idx, -0.5);
      return (
        <text
          key={idx}
          x={pos.x}
          y={30}
          textAnchor="middle"
          fill="white"
          fontSize="14"
          fontWeight="bold"
        >
          {layer.name}
        </text>
      );
    });
  };

  // Render gradients
  const renderGradientDefs = () => (
    <defs>
      {NETWORK.layers.map((layer, idx) => (
        <radialGradient key={idx} id={`gradient-${idx}`}>
          <stop offset="0%" stopColor={layer.color} />
          <stop offset="100%" stopColor={`${layer.color}88`} />
        </radialGradient>
      ))}
    </defs>
  );

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🧠 Neural Network Animation
          </h1>
          <p className="text-gray-400">
            Visualizing Forward & Backward Propagation
          </p>
        </div>

        {/* Control Panel */}
        <div className="bg-gray-800/50 rounded-xl p-6 mb-8 backdrop-blur-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            {/* Input Controls */}
            <div className="flex items-center gap-4">
              <div>
                <label className="text-gray-400 text-sm block mb-1">Input (XOR)</label>
                <div className="flex gap-2">
                  <select
                    className="bg-gray-700 text-white rounded px-3 py-2"
                    value={`${input[0]},${input[1]}`}
                    onChange={(e) => {
                      const [x1, x2] = e.target.value.split(',').map(Number);
                      setInput([x1, x2]);
                      setTarget(x1 ^ x2);
                      setActivations([[{ value: x1, preActivation: x1 }, { value: x2, preActivation: x2 }]]);
                    }}
                  >
                    <option value="0,0">0 XOR 0 = 0</option>
                    <option value="0,1">0 XOR 1 = 1</option>
                    <option value="1,0">1 XOR 0 = 1</option>
                    <option value="1,1">1 XOR 1 = 0</option>
                  </select>
                </div>
              </div>
              
              <div>
                <label className="text-gray-400 text-sm block mb-1">Target</label>
                <div className="bg-gray-700 text-green-400 rounded px-4 py-2 font-mono">
                  {target}
                </div>
              </div>
              
              {loss !== null && (
                <div>
                  <label className="text-gray-400 text-sm block mb-1">Loss (MSE)</label>
                  <div className="bg-gray-700 text-red-400 rounded px-4 py-2 font-mono">
                    {loss.toFixed(4)}
                  </div>
                </div>
              )}
              
              <div>
                <label className="text-gray-400 text-sm block mb-1">Epoch</label>
                <div className="bg-gray-700 text-yellow-400 rounded px-4 py-2 font-mono">
                  {epoch}
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <button
                onClick={initializeWeights}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg transition-colors"
              >
                Reset
              </button>
              
              {mode === 'intro' && (
                <button
                  onClick={() => { setMode('forward'); setStep(0); }}
                  className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
                >
                  Start Forward →
                </button>
              )}
              
              {mode === 'forward' && (
                <button
                  onClick={forwardStep}
                  className="px-6 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors"
                >
                  Forward Step →
                </button>
              )}
              
              {mode === 'forward-complete' && (
                <button
                  onClick={() => { setMode('backward'); setStep(NETWORK.layers.length - 1); }}
                  className="px-6 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg transition-colors"
                >
                  Start Backward ←
                </button>
              )}
              
              {mode === 'backward' && (
                <button
                  onClick={backwardStep}
                  className="px-6 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors"
                >
                  ← Backward Step
                </button>
              )}
              
              {mode === 'backward-complete' && (
                <>
                  <button
                    onClick={() => { updateWeights(); resetForNewIteration(); }}
                    className="px-6 py-2 bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors"
                  >
                    Update Weights & Train Again
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Network Visualization */}
        <div className="bg-gray-800/30 rounded-xl p-4 backdrop-blur-sm mb-8">
          <svg ref={svgRef} viewBox="0 0 900 500" className="w-full h-auto">
            {renderGradientDefs()}
            {renderConnections()}
            {renderNeurons()}
            {renderLayerLabels()}
          </svg>
        </div>

        {/* Information Panel */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Forward Propagation */}
          <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur-sm">
            <h3 className="text-xl font-bold text-green-400 mb-4">
              ➡️ Forward Propagation
            </h3>
            <div className="text-gray-300 space-y-2 font-mono text-sm">
              <p>For each layer l:</p>
              <p className="pl-4">z<sup>(l)</sup> = W<sup>(l)</sup> · a<sup>(l-1)</sup> + b</p>
              <p className="pl-4">a<sup>(l)</sup> = σ(z<sup>(l)</sup>)</p>
              <p className="mt-4 text-gray-400">
                Activation flows from input to output, computing weighted sums
                and applying non-linear activations.
              </p>
            </div>
          </div>

          {/* Backward Propagation */}
          <div className="bg-gray-800/50 rounded-xl p-6 backdrop-blur-sm">
            <h3 className="text-xl font-bold text-orange-400 mb-4">
              ⬅️ Backward Propagation
            </h3>
            <div className="text-gray-300 space-y-2 font-mono text-sm">
              <p>Output layer:</p>
              <p className="pl-4">δ<sup>(L)</sup> = ∇L · σ'(z<sup>(L)</sup>)</p>
              <p className="mt-2">Hidden layers:</p>
              <p className="pl-4">δ<sup>(l)</sup> = (W<sup>(l+1)</sup>)<sup>T</sup> · δ<sup>(l+1)</sup> ⊙ σ'(z<sup>(l)</sup>)</p>
              <p className="mt-4 text-gray-400">
                Gradients flow backward, computing how much each weight
                contributed to the error.
              </p>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="mt-8 text-center text-gray-400 text-sm">
          <div className="flex justify-center gap-8">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-400"></div>
              <span>Input Layer</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-purple-400"></div>
              <span>Hidden Layers</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-green-400"></div>
              <span>Output Layer</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-1 bg-indigo-500"></div>
              <span>Weights</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
