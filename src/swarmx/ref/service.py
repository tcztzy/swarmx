"""Bounded, read-only access to configured objective reference sources."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MAX_QUERY_CHARS = 256
MAX_SEARCH_RESULTS = 20
MAX_PAGE_CHARS = 32_000
MAX_ITEM_BYTES = 2 * 1024 * 1024
MAX_PATH_CHARS = 4_096
MAX_METADATA_CHARS = 2_048
MAX_NETWORK_BYTES = 256 * 1024
MAX_ESTIMATED_MATCHES = 2_147_483_647
NETWORK_TIMEOUT_SECONDS = 10
ZOTERO_API_URL = "http://127.0.0.1:23119/api"
SOURCE_IDS = frozenset({"zim", "zotero"})

JsonFetcher = Callable[[str], tuple[object, Mapping[str, str]]]


class ReferenceBackend(Protocol):
    def status(self) -> dict[str, Any]: ...

    def search(self, query: str, limit: int) -> dict[str, Any]: ...

    def get(self, path: str, max_chars: int) -> dict[str, Any]: ...


class ReferenceService:
    """Strict operation dispatcher shared by MCP and unit tests."""

    def __init__(self, backends: Mapping[str, ReferenceBackend]) -> None:
        if (
            not backends
            or set(backends) - SOURCE_IDS
            or not all(isinstance(source, str) for source in backends)
        ):
            raise ValueError("Reference sources are invalid.")
        self._backends = dict(backends)

    def handle(self, request: object) -> dict[str, Any]:
        if not isinstance(request, Mapping) or not all(
            isinstance(key, str) for key in request
        ):
            raise ValueError("Reference request must be an object.")
        operation = request.get("operation")
        if operation == "status":
            _require_fields(request, {"operation"}, {"source"})
            requested_source = request.get("source")
            source_ids = (
                [self._source_id(requested_source)]
                if requested_source is not None
                else list(self._backends)
            )
            sources = [
                {"id": source_id, **self._backends[source_id].status()}
                for source_id in source_ids
            ]
            result: dict[str, Any] = {
                "operation": "status",
                "sources": sources,
            }
            zim = next((source for source in sources if source["id"] == "zim"), None)
            if zim is not None and isinstance(zim.get("fileSize"), int):
                result["source"] = {
                    key: zim[key]
                    for key in (
                        "fileName",
                        "fileSize",
                        "title",
                        "language",
                        "date",
                        "description",
                    )
                }
            return result
        if operation == "search":
            _require_fields(request, {"operation", "query"}, {"limit", "source"})
            query = _bounded_string(request.get("query"), "query", MAX_QUERY_CHARS)
            limit = _bounded_integer(
                request.get("limit", 10), "limit", 1, MAX_SEARCH_RESULTS
            )
            source_id = self._selected_source(request.get("source"))
            result = self._backends[source_id].search(query, limit)
            matches = result.get("matches")
            if not isinstance(matches, list):
                raise ValueError("Reference source returned invalid search results.")
            return {
                "operation": "search",
                "source": source_id,
                **result,
                "matches": [{"source": source_id, **match} for match in matches],
            }
        if operation == "get":
            _require_fields(request, {"operation", "path"}, {"maxChars", "source"})
            path = _bounded_string(request.get("path"), "path", MAX_PATH_CHARS)
            max_chars = _bounded_integer(
                request.get("maxChars", MAX_PAGE_CHARS),
                "maxChars",
                1,
                MAX_PAGE_CHARS,
            )
            source_id = self._selected_source(request.get("source"))
            return {
                "operation": "get",
                "source": source_id,
                **self._backends[source_id].get(path, max_chars),
            }
        raise ValueError("Reference operation must be status, search, or get.")

    def _selected_source(self, value: object) -> str:
        if value is None:
            if "zim" not in self._backends:
                raise ValueError("Reference source must be selected.")
            return "zim"
        return self._source_id(value)

    def _source_id(self, value: object) -> str:
        if not isinstance(value, str) or value not in SOURCE_IDS:
            raise ValueError("Reference source is invalid.")
        if value not in self._backends:
            raise ValueError("Reference source is unavailable.")
        return value


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
            saved_stdout_fd = os.dup(1)
            try:
                sys.stdout.flush()
                # libzim may emit language-index diagnostics to fd 1; MCP owns stdout.
                os.dup2(2, 1)
                archive = Archive(str(resolved))
            finally:
                sys.stdout.flush()
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
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
            "kind": "zim",
            "name": self._path.name,
            "fileName": self._path.name,
            "fileSize": self._size,
            "title": self._metadata("Title"),
            "language": self._metadata("Language"),
            "date": self._metadata("Date"),
            "description": self._metadata("Description"),
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
            "estimatedMatches": _bounded_estimate(estimated),
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


class ZoteroReferenceBackend:
    """Read-only bibliographic metadata from Zotero Desktop's local API."""

    def __init__(self, *, fetch_json: JsonFetcher | None = None) -> None:
        self._fetch_json = fetch_json or _fetch_json
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "kind": "zotero",
            "name": "Zotero",
            "endpoint": f"{ZOTERO_API_URL}/",
        }

    def search(self, query: str, limit: int) -> dict[str, Any]:
        url = (
            f"{ZOTERO_API_URL}/users/0/items/top?"
            f"{urlencode({'q': query, 'limit': limit})}"
        )
        with self._lock:
            payload, headers = self._fetch_json(url)
        if not isinstance(payload, list):
            raise ValueError("Zotero returned an invalid response.")
        matches = [
            match
            for candidate in payload
            if (match := _zotero_match(candidate)) is not None
        ][:limit]
        estimated = _integer_header(headers, "Total-Results")
        return {
            "query": query,
            "mode": "zotero",
            "estimatedMatches": max(len(matches), estimated or 0),
            "matches": matches,
        }

    def get(self, path: str, max_chars: int) -> dict[str, Any]:
        if re.fullmatch(r"[23456789ABCDEFGHIJKLMNPQRSTUVWXYZ]{8}", path) is None:
            raise ValueError("Zotero item key is invalid.")
        with self._lock:
            payload, _headers = self._fetch_json(
                f"{ZOTERO_API_URL}/users/0/items/{path}"
            )
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("data"), Mapping
        ):
            raise ValueError("Zotero item was not found.")
        data = payload["data"]
        if data.get("itemType") in {"attachment", "note", "annotation"}:
            raise ValueError("Zotero attachment and note content is unavailable.")
        title = _bounded_optional_output(data.get("title")) or "Untitled Zotero item"
        text = _zotero_item_text(data, title)
        result: dict[str, Any] = {
            "path": path,
            "title": title,
            "mimeType": "text/plain",
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
        }
        item_url = _safe_result_url(data.get("url"))
        if item_url is not None:
            result["url"] = item_url
        return result


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


def _fetch_json(url: str) -> tuple[object, Mapping[str, str]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "SwarmX Reference",
            "Zotero-API-Version": "3",
        },
        method="GET",
    )
    try:
        opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
        with opener.open(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            content = response.read(MAX_NETWORK_BYTES + 1)
            if len(content) > MAX_NETWORK_BYTES:
                raise ValueError("Reference network response exceeds the safe limit.")
            payload = json.loads(content.decode("utf-8"))
            headers = {str(key): str(value) for key, value in response.headers.items()}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Reference network source is unavailable.") from error
    return payload, headers


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None


def _safe_result_url(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > MAX_PATH_CHARS:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "\x00" in value
    ):
        return None
    return value


def _bounded_optional_output(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = _normalize_text(html_to_text(value.replace("\x00", "")))
    return normalized[:MAX_METADATA_CHARS] or None


def _zotero_match(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("data"), Mapping):
        return None
    data = value["data"]
    key = value.get("key")
    if (
        not isinstance(key, str)
        or re.fullmatch(r"[23456789ABCDEFGHIJKLMNPQRSTUVWXYZ]{8}", key) is None
        or data.get("itemType") in {"attachment", "note", "annotation"}
    ):
        return None
    title = _bounded_optional_output(data.get("title")) or "Untitled Zotero item"
    result = {"path": key, "title": title}
    snippet = _bounded_optional_output(data.get("abstractNote"))
    item_url = _safe_result_url(data.get("url"))
    if snippet is not None:
        result["snippet"] = snippet
    if item_url is not None:
        result["url"] = item_url
    return result


def _zotero_item_text(data: Mapping[object, object], title: str) -> str:
    lines = [title]
    creators = data.get("creators")
    creator_names: list[str] = []
    if isinstance(creators, list):
        for creator in creators[:50]:
            if not isinstance(creator, Mapping):
                continue
            name = _bounded_optional_output(creator.get("name"))
            if name is None:
                name = _bounded_optional_output(
                    " ".join(
                        part
                        for part in (creator.get("firstName"), creator.get("lastName"))
                        if isinstance(part, str)
                    )
                )
            if name is not None:
                creator_names.append(name)
    if creator_names:
        lines.append(f"Creators: {', '.join(creator_names)}")
    for label, field in (
        ("Date", "date"),
        ("Publication", "publicationTitle"),
        ("DOI", "DOI"),
        ("URL", "url"),
        ("Abstract", "abstractNote"),
    ):
        value = _bounded_optional_output(data.get(field))
        if value is not None:
            lines.append(f"{label}: {value}")
    return _normalize_text("\n".join(lines))


def _integer_header(headers: Mapping[str, str], name: str) -> int | None:
    value = next(
        (value for key, value in headers.items() if key.lower() == name.lower()), None
    )
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return _bounded_estimate(parsed)


def _bounded_estimate(value: int) -> int:
    return max(0, min(value, MAX_ESTIMATED_MATCHES))


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
