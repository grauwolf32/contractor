const FULL_TURN = Math.PI * 2;

function point(x: number, y: number, move: boolean): string {
  return `${move ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
}

function spiralRing(index: number): string {
  const radius = 22 + index * 20;
  const points = 72;

  return Array.from({ length: points + 1 }, (_, pointIndex) => {
    const progress = pointIndex / points;
    const angle = progress * FULL_TURN;
    const expandingRadius = radius + progress * 12;
    return point(
      Math.cos(angle) * expandingRadius,
      Math.sin(angle) * expandingRadius,
      pointIndex === 0,
    );
  }).join(" ");
}

function spiralSpoke(index: number): string {
  const phase = (index / 18) * FULL_TURN;
  const points = 48;

  return Array.from({ length: points + 1 }, (_, pointIndex) => {
    const progress = pointIndex / points;
    const radius = 18 + progress * 306;
    const angle = phase + (1 - progress) * 1.18;
    return point(
      Math.cos(angle) * radius,
      Math.sin(angle) * radius,
      pointIndex === 0,
    );
  }).join(" ");
}

const rings = Array.from({ length: 15 }, (_, index) => spiralRing(index));
const spokes = Array.from({ length: 18 }, (_, index) => spiralSpoke(index));

function Spiral({ position }: { position: "near" | "far" }) {
  return (
    <svg
      className={`login-spiral login-spiral-${position}`}
      viewBox="-340 -340 680 680"
      aria-hidden="true"
      focusable="false"
    >
      <g className="login-spiral-rings">
        {rings.map((path, index) => (
          <path d={path} key={`ring-${index}`} />
        ))}
      </g>
      <g className="login-spiral-spokes">
        {spokes.map((path, index) => (
          <path
            className={index % 6 === 0 ? "login-spiral-accent" : undefined}
            d={path}
            key={`spoke-${index}`}
          />
        ))}
      </g>
    </svg>
  );
}

export function LoginBackdrop() {
  return (
    <>
      <Spiral position="near" />
      <Spiral position="far" />
    </>
  );
}
