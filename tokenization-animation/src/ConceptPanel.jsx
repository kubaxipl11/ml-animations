import React, { useState } from 'react';

export default function ConceptPanel() {
    const [text, setText] = useState("The quick brown fox jumps.");

    // Character Tokenization
    const charTokens = text.split('').map(c => `"${c}"`);

    // Word Tokenization
    const wordTokens = text.split(/\s+/).filter(w => w.length > 0).map(w => `"${w}"`);

    return (
        <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
            <div className="max-w-3xl w-full text-center mb-8">
                <h2 className="text-3xl font-bold text-purple-400 mb-4">Character vs Word Tokenization</h2>
                <p className="text-lg text-slate-300 leading-relaxed">
                    Computers only understand numbers. We must chop text into chunks (Tokens).
                    <br />
                    But how big should the chunks be?
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-12 w-full max-w-5xl items-start">
                {/* Controls */}
                <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                    <h3 className="font-bold text-white mb-6">Input Text</h3>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        className="w-full h-32 bg-slate-900 border border-slate-600 rounded p-4 text-white font-mono text-lg"
                        placeholder="Type something..."
                    />
                    <div className="mt-4 text-sm text-slate-400">
                        Try typing: "Unbelievably" or "Micro-transaction"
                    </div>
                </div>

                {/* Visualization */}
                <div className="space-y-8">
                    {/* Character Level */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-bold text-white">1. Character Level</h3>
                            <span className="text-xs bg-slate-700 px-2 py-1 rounded text-slate-300">{charTokens.length} Tokens</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {charTokens.map((t, i) => (
                                <span key={i} className="px-2 py-1 bg-purple-900/50 border border-purple-500/50 rounded text-purple-200 font-mono text-sm">
                                    {t}
                                </span>
                            ))}
                        </div>
                        <p className="text-xs text-slate-500 mt-3">
                            <strong>Pros:</strong> Small vocabulary (256 chars). No unknown words.
                            <br />
                            <strong>Cons:</strong> Sequences are very long. Hard to learn meaning.
                        </p>
                    </div>

                    {/* Word Level */}
                    <div className="bg-slate-800 p-6 rounded-xl border border-slate-700">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-bold text-white">2. Word Level</h3>
                            <span className="text-xs bg-slate-700 px-2 py-1 rounded text-slate-300">{wordTokens.length} Tokens</span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {wordTokens.map((t, i) => (
                                <span key={i} className="px-2 py-1 bg-fuchsia-900/50 border border-fuchsia-500/50 rounded text-fuchsia-200 font-mono text-sm">
                                    {t}
                                </span>
                            ))}
                        </div>
                        <p className="text-xs text-slate-500 mt-3">
                            <strong>Pros:</strong> Short sequences. Meaningful chunks.
                            <br />
                            <strong>Cons:</strong> Huge vocabulary (1M+ words). "Unbelievably" is totally different from "Believable".
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}
