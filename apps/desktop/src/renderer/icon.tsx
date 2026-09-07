const paths = {
  settings:
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M12 2v3 M12 19v3 M2 12h3 M19 12h3 M5 5l2 2 M17 17l2 2 M5 19l2-2 M17 7l2-2",
  graph: "M4 4h5v5H4z M15 15h5v5h-5z M15 4h5v5h-5z M9 6h6 M6 9v8h9",
  image: "M3 4h18v16H3z M3 16l6-6 4 4 3-3 5 5 M17 7h.01",
  play: "M7 4l14 8-14 8z",
  download: "M12 3v12 M7 10l5 5 5-5 M4 17v4h16v-4",
  swarm: "M4 4h6v6H4z M14 4h6v6h-6z M4 14h6v6H4z M14 14h6v6h-6z",
  compose:
    "M12 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-7 M16 3l5 5 M10 14l1-5 7-7 4 4-7 7z",
  search: "M21 21l-5-5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  folder: "M3 5h6l2 3h10v11H3z",
  sidebar: "M3 4h18v16H3z M9 4v16",
  refresh: "M20 7v5h-5 M4 17v-5h5 M6 6a8 8 0 0 1 13 3 M18 18a8 8 0 0 1-13-3",
  trace: "M4 5h16 M8 12h12 M12 19h8 M4 4v9h4 M8 11v9h4",
  plus: "M12 5v14 M5 12h14",
  close: "M6 6l12 12 M6 18L18 6",
  arrowUp: "M12 19V5 M5 12l7-7 7 7",
  arrowDown: "M12 5v14 M5 12l7 7 7-7",
  stop: "M6 6h12v12H6z",
  copy: "M8 8h12v12H8z M16 8V4H4v12h4",
  check: "M5 12l4 4L19 6",
  chevron: "M9 5l7 7-7 7",
  code: "M8 6l-6 6 6 6 M16 6l6 6-6 6 M14 4l-4 16",
  book: "M12 5v15 M12 5C9 3 5 3 2 4v15c4-1 7-1 10 1 3-2 6-2 10-1V4c-3-1-7-1-10 1",
} as const;

export function Icon({
  name,
  className = "size-4",
}: {
  name: keyof typeof paths;
  className?: string;
}) {
  return (
    <svg
      aria-hidden="true"
      className={`shrink-0 ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={paths[name]} />
    </svg>
  );
}
