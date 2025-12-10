"use client";

interface LayerSelectorProps {
  numLayers: number;
  numHeads: number;
  selectedLayer: number;
  selectedHead: number;
  availableLayers: number[];
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
}

export function LayerSelector({
  numLayers,
  numHeads,
  selectedLayer,
  selectedHead,
  availableLayers,
  onLayerChange,
  onHeadChange,
}: LayerSelectorProps) {
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-2">
          Layer ({availableLayers.length} available of {numLayers})
        </label>
        <div className="flex flex-wrap gap-2">
          {availableLayers.map((layer) => (
            <button
              key={layer}
              onClick={() => onLayerChange(layer)}
              className={`
                px-3 py-1 rounded-md text-sm font-medium transition-colors
                ${
                  selectedLayer === layer
                    ? "bg-blue-600 text-white"
                    : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                }
              `}
            >
              {layer}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-2">
          Attention Head (0-{numHeads - 1})
        </label>
        <input
          type="range"
          min={0}
          max={numHeads - 1}
          value={selectedHead}
          onChange={(e) => onHeadChange(parseInt(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-zinc-500 mt-1">
          <span>0</span>
          <span className="text-blue-400 font-medium">Head {selectedHead}</span>
          <span>{numHeads - 1}</span>
        </div>
      </div>

      <div className="text-xs text-zinc-500">
        <p>
          <strong>Tip:</strong> Earlier layers capture low-level features
          (edges, colors). Later layers capture high-level semantics (objects,
          actions).
        </p>
      </div>
    </div>
  );
}
