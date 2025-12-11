"use client";

interface LayerSelectorProps {
  numLayers: number;
  numHeads: number;
  selectedLayer: number;
  selectedHead: number;
  availableLayers: number[];
  availableHeads: number[];
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
}

export function LayerSelector({
  numLayers,
  numHeads,
  selectedLayer,
  selectedHead,
  availableLayers,
  availableHeads,
  onLayerChange,
  onHeadChange,
}: LayerSelectorProps) {
  // Check if we're showing averaged heads only
  const isAveragedOnly = availableHeads.length === 1 && availableHeads[0] === -1;

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

      {!isAveragedOnly && (
        <div>
          <label className="block text-sm font-medium text-zinc-400 mb-2">
            Attention Head ({availableHeads.length} available of {numHeads})
          </label>
          <div className="flex flex-wrap gap-2">
            {availableHeads.map((head) => (
              <button
                key={head}
                onClick={() => onHeadChange(head)}
                className={`
                  px-3 py-1 rounded-md text-sm font-medium transition-colors
                  ${
                    selectedHead === head
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-800 text-zinc-300 hover:bg-zinc-700"
                  }
                `}
              >
                {head === -1 ? "Avg" : head}
              </button>
            ))}
          </div>
        </div>
      )}

      {isAveragedOnly && (
        <div className="text-xs text-zinc-500 bg-zinc-800/50 px-3 py-2 rounded">
          Attention averaged across all {numHeads} heads for cleaner visualization
        </div>
      )}
    </div>
  );
}
