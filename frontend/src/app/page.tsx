"use client";

import { useState, useCallback } from "react";
import { AttentionHeatmap } from "@/components/AttentionHeatmap";
import { ImageUpload } from "@/components/ImageUpload";
import { AdapterSelector } from "@/components/AdapterSelector";

interface AttentionEntry {
  layer: number;
  head: number;
  attention: number[];
}

interface AnalysisResult {
  output_text: string;
  attention_data: AttentionEntry[];
  num_layers: number;
  num_heads: number;
  tokens: string[];
  vision_token_range: (number | null)[];
  image_size: number[];
  adapter_name: string | null;
  coordinates: [number, number] | null;
  generated_tokens: string[];
  coordinate_token_indices: number[];
  vision_grid: [number, number] | null;
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [query, setQuery] = useState("What action should I take?");
  const [adapter, setAdapter] = useState<string>("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setImage(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
  }, []);

  const handleAnalyze = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", image);
      formData.append("query", query);
      if (adapter) {
        formData.append("adapter_name", adapter);
      }

      const response = await fetch("http://localhost:9002/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData.detail || response.statusText;
        throw new Error(`Analysis failed: ${errorMsg}`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-900 text-zinc-100">
      <main className="px-6 py-4">
        {/* Controls Bar */}
        <div className="mb-6">
          <div className="flex flex-wrap gap-4 items-end">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-zinc-400 mb-1">
                Query
              </label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full rounded-md bg-zinc-800 px-3 py-2 text-zinc-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                placeholder="What action should I take?"
              />
            </div>

            <div className="min-w-[250px]">
              <label className="block text-sm font-medium text-zinc-400 mb-1">
                LoRA Adapter
              </label>
              <AdapterSelector value={adapter} onChange={setAdapter} />
            </div>

            <button
              onClick={handleAnalyze}
              disabled={!image || loading}
              className="rounded-md bg-blue-600 px-6 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          {error && (
            <div className="mt-4 rounded-md bg-red-900/50 px-4 py-2 text-red-200">
              {error}
            </div>
          )}
        </div>

        {/* Main Visualization Area */}
        <div>
          {!imageUrl && !result && (
            <ImageUpload onImageSelect={handleImageSelect} />
          )}

          {imageUrl && !result && (
            <div className="space-y-4">
              {!loading && (
                <div className="flex justify-between items-center">
                  <span className="text-sm text-zinc-400">Image loaded</span>
                  <button
                    onClick={() => { setImage(null); setImageUrl(null); }}
                    className="text-sm text-zinc-500 hover:text-zinc-300"
                  >
                    Clear
                  </button>
                </div>
              )}
              <div className="relative">
                <img
                  src={imageUrl}
                  alt="Selected"
                  className={`w-full rounded-lg ${loading ? "opacity-50" : ""}`}
                />
                {loading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded-lg">
                    <div className="bg-zinc-900/90 px-6 py-4 rounded-lg text-center">
                      <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-3"></div>
                      <p className="text-zinc-200 font-medium">Running inference...</p>
                      <p className="text-xs text-zinc-400 mt-1">Model warmup may take 30-60s on first request</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <div className="text-sm text-zinc-400">
                  <span className="text-zinc-500">Adapter:</span> {result.adapter_name || "Base"}
                  <span className="mx-2">•</span>
                  <span className="text-zinc-500">Target:</span> {result.coordinates ? `[${result.coordinates.join(", ")}]` : "None"}
                  <span className="mx-2">•</span>
                  <span className="text-zinc-500">Grid:</span> {result.vision_grid ? `${result.vision_grid[0]}×${result.vision_grid[1]}` : "?"}
                </div>
                <button
                  onClick={() => { setResult(null); }}
                  className="text-sm text-zinc-500 hover:text-zinc-300"
                >
                  New Image
                </button>
              </div>

              <AttentionHeatmap
                imageUrl={imageUrl!}
                attentionData={result.attention_data}
                coordinates={result.coordinates}
                visionGrid={result.vision_grid}
                imageSize={result.image_size as [number, number] | null}
              />

              <div className="rounded-md bg-zinc-800 px-4 py-3 font-mono text-sm whitespace-pre-wrap">
                {result.output_text}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
