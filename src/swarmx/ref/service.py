"""Bounded, read-only access to one offline ZIM reference archive."""

from __future__ import annotations

import re
import threading
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Protocol

MAX_QUERY_CHARS = 256
MAX_SEARCH_RESULTS = 20
MAX_PAGE_CHARS = 32_000
MAX_ITEM_BYTES = 2 * 1024 * 1024
MAX_PATH_CHARS = 4_096
MAX_METADATA_CHARS = 2_048


class ReferenceBackend(Protocol):
    def status(self) -> dict[str, Any]: ...

    def search(self, query: str, limit: int) -> dict[str, Any]: ...

    def get(self, path: str, max_chars: int) -> dict[str, Any]: ...


class ReferenceService:
    """Strict operation dispatcher shared by MCP and unit tests."""

    def __init__(self, backend: ReferenceBackend) -> None:
        self._backend = backend

    def handle(self, request: object) -> dict[str, Any]:
        if not isinstance(request, Mapping) or not all(
            isinstance(key, str) for key in request
        ):
            raise ValueError("Reference request must be an object.")
        operation = request.get("operation")
        if operation == "status":
            _require_fields(request, {"operation"})
            return {"operation": "status", **self._backend.status()}
        if operation == "search":
            _require_fields(request, {"operation", "query"}, {"limit"})
            query = _bounded_string(request.get("query"), "query", MAX_QUERY_CHARS)
            limit = _bounded_integer(
                request.get("limit", 10), "limit", 1, MAX_SEARCH_RESULTS
            )
            return {"operation": "search", **self._backend.search(query, limit)}
        if operation == "get":
            _require_fields(request, {"operation", "path"}, {"maxChars"})
            path = _bounded_string(request.get("path"), "path", MAX_PATH_CHARS)
            max_chars = _bounded_integer(
                request.get("maxChars", MAX_PAGE_CHARS),
                "maxChars",
                1,
                MAX_PAGE_CHARS,
            )
            return {"operation": "get", **self._backend.get(path, max_chars)}
        raise ValueError("Reference operation must be status, search, or get.")


class LibzimReferenceBackend:
    """Official python-libzim adapter with serialized search access."""

    def __init__(self, zim_path: str) -> None:
        resolved = Path(zim_path).expanduser().resolve()
        if len(str(resolved)) > MAX_PATH_CHARS or resolved.suffix.lower() != ".zim":
            raise ValueError("Reference source must be a local .zim file.")
        try:
            stat = resolved.stat()
        except OSError as error:
            raise ValueError("Reference source is unavailable.") from error
        if not resolved.is_file():
            raise ValueError("Reference source must be a regular .zim file.")
        from libzim.reader import Archive

        try:
            archive = Archive(str(resolved))
        except Exception as error:
            raise ValueError(
                "Reference source is not a readable ZIM archive."
            ) from error
        self._path = resolved
        self._size = stat.st_size
        self._archive = archive
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "source": {
                "fileName": self._path.name,
                "fileSize": self._size,
                "title": self._metadata("Title"),
                "language": self._metadata("Language"),
                "date": self._metadata("Date"),
                "description": self._metadata("Description"),
            }
        }

    def search(self, query: str, limit: int) -> dict[str, Any]:
        from libzim.search import Query, Searcher
        from libzim.suggestion import SuggestionSearcher

        with self._lock:
            try:
                search = Searcher(self._archive).search(Query().set_query(query))
                estimated = int(search.getEstimatedMatches())
                paths = list(search.getResults(0, min(estimated, limit)))
                mode = "full_text"
            except Exception:
                suggestion = SuggestionSearcher(self._archive).suggest(query)
                estimated = int(suggestion.getEstimatedMatches())
                paths = list(suggestion.getResults(0, min(estimated, limit)))
                mode = "suggestion"
            matches = [self._entry_summary(str(path)) for path in paths]
        return {
            "query": query,
            "mode": mode,
            "estimatedMatches": max(0, estimated),
            "matches": matches,
        }

    def get(self, path: str, max_chars: int) -> dict[str, Any]:
        with self._lock:
            try:
                entry = self._archive.get_entry_by_path(path)
                item = entry.get_item()
            except Exception as error:
                raise ValueError("Reference article was not found.") from error
            size = int(item.size)
            if size > MAX_ITEM_BYTES:
                raise ValueError("Reference article exceeds the safe read limit.")
            mime_type = str(item.mimetype)
            content = bytes(item.content)
            if len(content) > MAX_ITEM_BYTES:
                raise ValueError("Reference article exceeds the safe read limit.")
            decoded = content.decode("utf-8", errors="replace")
            text = html_to_text(decoded) if "html" in mime_type.lower() else decoded
            text = _normalize_text(text)
            truncated = len(text) > max_chars
            return {
                "path": str(entry.path),
                "title": _bounded_output(str(entry.title)),
                "mimeType": mime_type[:128],
                "text": text[:max_chars],
                "truncated": truncated,
            }

    def _entry_summary(self, path: str) -> dict[str, str]:
        try:
            entry = self._archive.get_entry_by_path(path)
        except Exception as error:
            raise ValueError("Reference search returned an invalid article.") from error
        return {"path": str(entry.path), "title": _bounded_output(str(entry.title))}

    def _metadata(self, key: str) -> str | None:
        try:
            value = self._archive.get_metadata(key)
        except Exception:
            return None
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return _bounded_output(str(value))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style", "noscript", "template"}:
            self._ignored_depth += 1
        elif self._ignored_depth == 0 and tag in {
            "p",
            "br",
            "div",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "tr",
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "template"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif self._ignored_depth == 0 and tag in {"p", "div", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0:
            self.parts.append(data)


def html_to_text(content: str) -> str:
    parser = _TextExtractor()
    parser.feed(content)
    parser.close()
    return unescape("".join(parser.parts))


def _normalize_text(value: str) -> str:
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _bounded_output(value: str) -> str:
    return value.replace("\x00", "")[:MAX_METADATA_CHARS]


def _bounded_string(value: object, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise ValueError(f"Reference {label} is invalid.")
    return value.strip()


def _bounded_integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"Reference {label} is invalid.")
    return value


def _require_fields(
    request: Mapping[object, object],
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    if set(request) - allowed or not required.issubset(request):
        raise ValueError("Reference request fields are invalid.")
