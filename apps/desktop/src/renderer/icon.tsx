import {
  ArrowDownIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon,
  ArrowRightIcon,
  ArrowTopRightOnSquareIcon,
  ArrowUpIcon,
  Bars3Icon,
  BookOpenIcon,
  CheckCircleIcon,
  CheckIcon,
  ChevronRightIcon,
  CodeBracketIcon,
  Cog6ToothIcon,
  DocumentDuplicateIcon,
  EyeIcon,
  FolderIcon,
  MagnifyingGlassIcon,
  PaperAirplaneIcon,
  PaperClipIcon,
  PencilSquareIcon,
  PhotoIcon,
  PlayIcon,
  PlusIcon,
  QueueListIcon,
  ShareIcon,
  Squares2X2Icon,
  StopIcon,
  XCircleIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";

const icons = {
  settings: Cog6ToothIcon,
  graph: ShareIcon,
  image: PhotoIcon,
  play: PlayIcon,
  download: ArrowDownTrayIcon,
  swarm: Squares2X2Icon,
  compose: PencilSquareIcon,
  search: MagnifyingGlassIcon,
  folder: FolderIcon,
  sidebar: Bars3Icon,
  refresh: ArrowPathIcon,
  trace: QueueListIcon,
  plus: PlusIcon,
  close: XMarkIcon,
  arrowUp: ArrowUpIcon,
  arrowDown: ArrowDownIcon,
  arrowRight: ArrowRightIcon,
  stop: StopIcon,
  copy: DocumentDuplicateIcon,
  check: CheckIcon,
  chevron: ChevronRightIcon,
  code: CodeBracketIcon,
  book: BookOpenIcon,
  eye: EyeIcon,
  external: ArrowTopRightOnSquareIcon,
  checkCircle: CheckCircleIcon,
  errorCircle: XCircleIcon,
  send: PaperAirplaneIcon,
  attachment: PaperClipIcon,
};

export function Icon({
  name,
  className = "size-4",
}: {
  name: keyof typeof icons;
  className?: string;
}) {
  const Component = icons[name];
  return <Component aria-hidden="true" className={`shrink-0 ${className}`} strokeWidth={1.5} />;
}
