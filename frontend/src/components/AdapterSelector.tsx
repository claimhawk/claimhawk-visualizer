"use client";

import { useEffect, useState } from "react";

interface Adapter {
  name: string;
  description: string;
  type: "router" | "expert";
  label?: number;
}

interface AdapterSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export function AdapterSelector({ value, onChange }: AdapterSelectorProps) {
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchAdapters(): Promise<void> {
      try {
        console.log("Fetching adapters...");
        const response = await fetch("http://localhost:9002/api/adapters");
        console.log("Response:", response.status);
        if (response.ok) {
          const data = await response.json();
          console.log("Adapters loaded:", data);
          setAdapters(data);
        }
      } catch (err) {
        console.error("Failed to fetch adapters:", err);
      } finally {
        setLoading(false);
      }
    }

    fetchAdapters();
  }, []);

  const experts = adapters.filter((a) => a.type === "expert");
  const router = adapters.find((a) => a.type === "router");

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-zinc-700 bg-zinc-800 px-4 py-2 text-zinc-100 focus:border-blue-500 focus:outline-none"
    >
      <option value="">Base Model (no adapter)</option>
      {loading ? (
        <option disabled>Loading adapters...</option>
      ) : (
        <>
          {router && (
            <optgroup label="Router">
              <option value={router.name}>
                {router.name} - {router.description}
              </option>
            </optgroup>
          )}
          <optgroup label="Experts">
            {experts.map((adapter) => (
              <option key={adapter.name} value={adapter.name}>
                {adapter.name} - {adapter.description}
              </option>
            ))}
          </optgroup>
        </>
      )}
    </select>
  );
}
