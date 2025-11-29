import React, { useState, useEffect } from 'react';
import { ArrowRight, RefreshCw } from 'lucide-react';

export default function BPEPanel() {
    const [word, setWord] = useState("unbelievably");
    const [tokens, setTokens] = useState(word.split(''));
    const [vocab, setVocab] = useState(new Set(word.split('')));
    const [merges, setMerges] = useState([]);

    // Reset when word changes
    useEffect(() => {
        setTokens(word.split(''));
        setVocab(new Set(word.split('')));
        setMerges([]);
    }, [word]);

    const findBestPair = (currentTokens) => {
        const pairs = {};
        for (let i = 0; i < currentTokens.length - 1; i++) {
            const pair = currentTokens[i] + currentTokens[i + 1];
            pairs[pair] = (pairs[pair] || 0) + 1;
        }

        // Find most frequent pair
        let bestPair = null;
        let maxCount = -1;

        Object.entries(pairs).forEach(([pair, count]) => {
            if (count > maxCount) {
                maxCount = count;
                bestPair = pair;
            }
        });

        return bestPair;
    };

    const performMerge = () => {
        const pair = findBestPair(tokens);
        if (!pair) return;

        const newTokens = [];
        let i = 0;
        while (i < tokens.length) {
            if (i < tokens.length - 1 && tokens[i] + tokens[i + 1] === pair) {
                newTokens.push(pair);
                i += 2;
            } else {
                newTokens.push(tokens[i]);
                i++;
            }
        }

        setTokens(newTokens);
        setVocab(prev => new Set([...prev, pair]));
        setMerges(prev => [...prev, pair]);
    };

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-fuchsia-400 mb-4">Byte Pair Encoding (BPE)</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    The "Goldilocks" solution. Start with characters, then iteratively <strong>merge</strong> the most frequent pairs.
                    <br />
                    This is what GPT-4 uses!
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-start">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Simulation</h3>

                    <div className="mb-6">
                        <label className="text-sm text-slate-400 mb-2 block">Target Word</label>
                        <input
                            type="text"
                            value={word}
                            onChange={(e) => setWord(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-600 rounded p-3 text-white font-mono text-lg"
                        />
                    </div>

                    <button
                        onClick={performMerge}
                        disabled={tokens.length <= 1}
                        className="w-full py-3 bg-fuchsia-600 hover:bg-fuchsia-500 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded-xl font-bold shadow-lg transition-all flex items-center justify-center gap-2"
                    >
                        <RefreshCw size={20} />
                        Perform Next Merge
                    </button>

                    <div className="mt-6">
                        <h4 className="font-bold text-slate-300 mb-2">Merge History:</h4>
                        <div className="space-y-1 text-sm font-mono text-slate-400">
                            {merges.length === 0 && <span className="italic">No merges yet.</span>}
                            {merges.map((m, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <span className="text-slate-600">{i + 1}.</span>
                                    <span className="text-fuchsia-400">"{m}"</span> added to vocab
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Visualization */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700 flex flex-col items-center">
                    <h3 className="font-bold text-white mb-8">Current Tokens</h3>

                    <div className="flex flex-wrap justify-center gap-2 mb-12">
                        {tokens.map((t, i) => (
                            <div key={i} className="relative group">
                                <span className="px-4 py-3 bg-gradient-to-br from-purple-900 to-fuchsia-900 border border-fuchsia-500/50 rounded-lg text-white font-mono text-xl shadow-lg block">
                                    {t}
                                </span>
                                <span className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-slate-500 font-mono">
                                    ID: {t.length}
                                </span>
                            </div>
                        ))}
                    </div>

                    <div className="w-full bg-slate-900 p-6 rounded-xl border border-fuchsia-500/30">
                        <div className="flex justify-between items-center mb-4">
                            <h4 className="font-bold text-slate-300">Vocabulary ({vocab.size})</h4>
                        </div>
                        <div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
                            {Array.from(vocab).map((v, i) => (
                                <span key={i} className={`px-2 py-1 rounded text-xs font-mono border ${merges.includes(v)
                                        ? 'bg-fuchsia-900/30 border-fuchsia-500/50 text-fuchsia-300'
                                        : 'bg-slate-800 border-slate-700 text-slate-400'
                                    }`}>
                                    "{v}"
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
