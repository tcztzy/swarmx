import { APP_ICON_URL } from "./app-icon-data.js";

export { APP_ICON_URL };

export function AppBrandIcon({ className }: { className?: string }) {
  return (
    <img className={className} src={APP_ICON_URL} alt="" aria-hidden="true" draggable={false} />
  );
}
