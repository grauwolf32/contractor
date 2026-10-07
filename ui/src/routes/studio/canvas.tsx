import { useRef, useState } from "react";
import type { BlockType } from "./document";
import type { StudioGraph, StudioNode } from "./graph";
import type { Problem } from "./validation";

export function StudioCanvas({
  graph,
  selected,
  onSelect,
  problems,
  onAdd,
  connect,
  onConnect,
}: {
  graph: StudioGraph;
  selected: string;
  onSelect: (node: StudioNode, showProperties?: boolean) => void;
  problems: Problem[];
  onAdd: (type: BlockType) => void;
  connect: string | undefined;
  onConnect: (node: StudioNode) => void;
}) {
  const [positions, setPositions] = useState<
    Record<string, { x: number; y: number }>
  >({});
  const [zoom, setZoom] = useState(1);
  const viewport = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    id: string;
    x: number;
    y: number;
    startX: number;
    startY: number;
  } | null>(null);
  const pan = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);
  const nodes = graph.nodes.map((node) => ({
    ...node,
    ...(positions[node.id] ?? {}),
  }));
  const width = Math.max(950, ...nodes.map((node) => node.x + 260));
  const height = Math.max(600, ...nodes.map((node) => node.y + 140));
  const move = (id: string, x: number, y: number) =>
    setPositions((current) => ({
      ...current,
      [id]: { x: Math.max(10, x), y: Math.max(20, y) },
    }));
  return (
    <section className="studio-canvas" aria-label="Definition graph">
      <div className="studio-canvas-tools">
        <button
          className="ui-btn"
          onClick={() => setZoom((value) => Math.max(0.4, value - 0.2))}
          aria-label="Zoom out"
        >
          −
        </button>
        <output aria-label="Canvas zoom">{Math.round(zoom * 100)}%</output>
        <button
          className="ui-btn"
          onClick={() => setZoom((value) => Math.min(1.6, value + 0.2))}
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          className="ui-btn"
          onClick={() => {
            setPositions({});
            setZoom(1);
            viewport.current?.scrollTo({ top: 0, left: 0 });
          }}
        >
          Reset layout
        </button>
      </div>
      <p className="studio-canvas-help">
        {connect
          ? "Choose the destination stage. Escape cancels the connection."
          : "Select a block to edit. Drag its handle to arrange; arrow keys move a focused handle."}
      </p>
      <div
        ref={viewport}
        className="studio-viewport"
        tabIndex={0}
        aria-label="Scrollable graph canvas"
        onDragOver={(event) => {
          if (
            event.dataTransfer.types.includes(
              "application/contractor-studio-block",
            )
          )
            event.preventDefault();
        }}
        onDrop={(event) => {
          event.preventDefault();
          const type = event.dataTransfer.getData(
            "application/contractor-studio-block",
          );
          if (
            [
              "stage",
              "input",
              "output",
              "parameter",
              "role",
              "toolset",
              "skill",
            ].includes(type)
          )
            onAdd(type as BlockType);
        }}
        onPointerDown={(event) => {
          if ((event.target as HTMLElement).closest("button")) return;
          pan.current = {
            x: event.clientX,
            y: event.clientY,
            left: event.currentTarget.scrollLeft,
            top: event.currentTarget.scrollTop,
          };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!pan.current) return;
          event.currentTarget.scrollLeft =
            pan.current.left - (event.clientX - pan.current.x);
          event.currentTarget.scrollTop =
            pan.current.top - (event.clientY - pan.current.y);
        }}
        onPointerUp={() => {
          pan.current = null;
        }}
        onPointerCancel={() => {
          pan.current = null;
        }}
      >
        <div style={{ width: width * zoom, height: height * zoom }}>
          <div
            className="studio-world"
            style={{ width, height, transform: `scale(${zoom})` }}
          >
            <svg
              width={width}
              height={height}
              className="studio-wires"
              aria-hidden="true"
            >
              {graph.edges.map((edge, index) => {
                const from = nodes.find((node) => node.id === edge.from)!,
                  to = nodes.find((node) => node.id === edge.to)!;
                const sx = from.x + 218,
                  sy = from.y + 50,
                  tx = to.x,
                  ty = to.y + 50;
                const bend = Math.max(45, Math.abs(tx - sx) * 0.4);
                return (
                  <g key={index} data-edge={edge.type}>
                    <path
                      d={`M ${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`}
                    />
                    <circle cx={tx} cy={ty} r={3} />
                    <text x={(sx + tx) / 2} y={(sy + ty) / 2 - 8}>
                      {edge.label}
                    </text>
                  </g>
                );
              })}
            </svg>
            {nodes.map((node) => {
              const issues = problems.filter((problem) =>
                node.path.every((part, index) => problem.path[index] === part),
              );
              return (
                <div
                  key={node.id}
                  className="studio-node"
                  data-selected={node.id === selected}
                  data-error={issues.some(
                    (issue) => issue.severity === "error",
                  )}
                  style={{ left: node.x, top: node.y }}
                >
                  <button
                    className="studio-node-handle"
                    aria-label={`Move ${node.title}`}
                    title="Drag or use arrow keys"
                    onPointerDown={(event) => {
                      event.stopPropagation();
                      onSelect(node, false);
                      drag.current = {
                        id: node.id,
                        x: node.x,
                        y: node.y,
                        startX: event.clientX,
                        startY: event.clientY,
                      };
                      event.currentTarget.setPointerCapture(event.pointerId);
                    }}
                    onPointerMove={(event) => {
                      if (!drag.current || drag.current.id !== node.id) return;
                      move(
                        node.id,
                        drag.current.x +
                          (event.clientX - drag.current.startX) / zoom,
                        drag.current.y +
                          (event.clientY - drag.current.startY) / zoom,
                      );
                    }}
                    onPointerUp={() => {
                      drag.current = null;
                    }}
                    onPointerCancel={() => {
                      drag.current = null;
                    }}
                    onKeyDown={(event) => {
                      const vector = {
                        ArrowLeft: [-1, 0],
                        ArrowRight: [1, 0],
                        ArrowUp: [0, -1],
                        ArrowDown: [0, 1],
                      }[event.key];
                      if (!vector) return;
                      event.preventDefault();
                      move(
                        node.id,
                        node.x + vector[0]! * 20,
                        node.y + vector[1]! * 20,
                      );
                    }}
                  >
                    ⠿ <span>{node.type}</span>
                  </button>
                  <button
                    className="studio-node-select"
                    aria-pressed={node.id === selected}
                    onClick={() =>
                      connect && node.type === "stage"
                        ? onConnect(node)
                        : onSelect(node)
                    }
                  >
                    <strong>{node.title}</strong>
                    <span>{node.subtitle}</span>
                    {issues.length ? (
                      <small>
                        {issues.length}{" "}
                        {issues.some((issue) => issue.severity === "error")
                          ? "error(s)"
                          : "warning(s)"}
                      </small>
                    ) : null}
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <details className="studio-edge-list">
        <summary>Connections ({graph.edges.length})</summary>
        <ul>
          {graph.edges.map((edge, index) => (
            <li key={index}>
              {nodes.find((node) => node.id === edge.from)?.title} →{" "}
              {nodes.find((node) => node.id === edge.to)?.title}: {edge.label}
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
