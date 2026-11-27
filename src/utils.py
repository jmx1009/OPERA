import hashlib
import os
import re
import time
import json
from typing import Any, Callable
from dotenv import load_dotenv

#* Configure
from .config import training_config, validate_model
config = training_config()

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _openai_http_client():
    import httpx

    # httpx reads HTTP_PROXY/HTTPS_PROXY by default. A stale or protocol-mismatched
    # proxy surfaces as opaque OpenAI "Connection error" failures, so local runs use
    # direct connections unless proxy trust is explicitly requested.
    return httpx.Client(trust_env=_env_flag("OPERA_API_TRUST_ENV", False))


def _coerce_content_to_text(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"} and item.get("text") is not None:
                    parts.append(str(item.get("text", "")))
                elif item.get("output_text") is not None:
                    parts.append(str(item.get("output_text", "")))
                elif "content" in item:
                    parts.append(_coerce_content_to_text(item.get("content")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        if content.get("type") in {"text", "output_text"} and content.get("text") is not None:
            return str(content.get("text", ""))
        for key in ("output_text", "text", "content"):
            if key in content:
                return _coerce_content_to_text(content.get(key))
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def _normalize_messages(prompt: Any) -> list[dict]:
    if isinstance(prompt, list):
        normalized = []
        for msg in prompt:
            if not isinstance(msg, dict):
                normalized.append({"role": "user", "content": _coerce_content_to_text(msg)})
                continue

            normalized.append({
                "role": msg.get("role", "user"),
                "content": _coerce_content_to_text(msg.get("content")),
            })
        return normalized

    return [{"role": "user", "content": _coerce_content_to_text(prompt)}]


def _get_llm_request_limit(name: str, default: int) -> int:
    params_cfg = getattr(config, "params", None)
    value = getattr(params_cfg, name, None) if params_cfg is not None else None
    if value is None:
        value = getattr(config, name, default)

    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _build_client(model: str, service: str | None):
    from openai import OpenAI

    validate_model(model, service)
    load_dotenv()
    return OpenAI(http_client=_openai_http_client(), max_retries=0)


def _json_digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def call_llm_and_parse_with_retry(
    model: str,
    service: str | None = None,
    prompt: str | None = None,
    parse_fn: Callable[..., Any] = None,
    temperature: float = 0.7,        # sampling temperature
    max_retry: int = 3,
    sleep_sec: float = 2,
    verbose: bool = True,
    log_header: str | None = None,
    error_message: str | None = None,
    response_records: list | None = None,
) -> Any:
    """Request a GPT-4o chat completion and parse its text with the supplied parser."""
    current_prompt = prompt
    request_timeout = _get_llm_request_limit("llm_timeout_sec", 120)
    request_max_tokens = _get_llm_request_limit("llm_max_output_tokens", 15000)

    def _send_request(messages):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=request_max_tokens,
            timeout=request_timeout,
        )
        if not completion.choices:
            raise ValueError("The chat completion contains no choices.")
        return completion

    with _build_client(model, service) as client:
        # Retry loop
        for attempt in range(1, max_retry + 1):

            if log_header is not None:
                if verbose: print(log_header)
            try:
                t0 = time.time()
                if verbose: print(f"[Attempt {attempt}/{max_retry}]\n")

                messages = _normalize_messages(current_prompt)
                completion = _send_request(messages)
                raw_text = completion.choices[0].message.content or ""
                if raw_text is None or not str(raw_text).strip():
                    raise RuntimeError("Empty LLM response")
                if verbose:
                    print(raw_text)
                resp_time = time.time() - t0

                if verbose: print(f"Done in {resp_time:.2f}s")

                result = parse_fn(raw_text)
                if response_records is not None:
                    response_records.append({
                        "requested_model": model,
                        "returned_model": getattr(completion, "model", None),
                        "completion_id": getattr(completion, "id", None),
                        "created": getattr(completion, "created", None),
                        "messages_sha256": _json_digest(messages),
                        "response_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                        "parsed_sha256": _json_digest(result),
                    })
                return result

            except Exception as err:
                err_text = str(err)

                # final attempt → raise
                if attempt == max_retry:
                    raise RuntimeError(error_message or
                                    f"\nFailed to parse results from LLM after {max_retry} attempts") from err

                # If the model returned non-JSON text, tighten the next retry prompt.
                if isinstance(current_prompt, str) and (
                    "No JSON object found" in err_text
                    or "No JSON array found" in err_text
                    or "No valid JSON array of objects found" in err_text
                    or "parsed JSON" in err_text
                ):
                    current_prompt = (
                        current_prompt
                        + "\n\nIMPORTANT: Return only valid JSON in exactly the requested format. "
                          "Omit Markdown fences and surrounding prose. "
                          "Keep each string field compact, avoid long multiline excerpts, and ensure the response ends with the closing ] or }."
                    )
                elif isinstance(current_prompt, str) and (
                    "No valid code fence" in err_text
                    or "No valid ```...``` block" in err_text
                    or "No fenced block" in err_text
                ):
                    if "code fence" in err_text:
                        current_prompt = (
                            current_prompt
                            + "\n\nFORMAT RECOVERY: Your entire response must be exactly one Markdown fenced Python code block. "
                              "The first line must be ```python and the final line must be ```. "
                              "Do not include explanations, headings, comments outside the code block, or any text before or after the fence."
                        )
                    else:
                        current_prompt = (
                            current_prompt
                            + "\n\nFORMAT RECOVERY: Your entire response must be exactly one Markdown fenced block. "
                              "The first line must be ``` and the final line must be ```. "
                              "Do not include explanations, headings, markdown outside the block, or any text before or after the fence."
                        )
                elif isinstance(current_prompt, str) and "Empty LLM response" in err_text:
                    current_prompt = (
                        current_prompt
                        + "\n\nIMPORTANT: Your previous response was empty. Return the requested artifact now, "
                          "using exactly the output format specified above and no extra commentary."
                    )

                # exponential back-off
                backoff = sleep_sec * (2 ** (attempt - 1))
                print(f"\nFailed to parse results on attempt {attempt}. \nError: {err}. \nRetrying in {backoff:.1f}s …")
                time.sleep(backoff)


def save_log_data(data, data_path):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    suffix = os.path.splitext(data_path)[1]
    if data is None or suffix not in {".json", ".py", ".txt"}:
        return
    with open(data_path, "w", encoding="utf-8") as stream:
        if suffix == ".json":
            json.dump(data, stream, indent=2, ensure_ascii=False)
        else:
            stream.write(data)


def cal_time_cost(start_time, phase_name):
    """
    Calculate the the duration of a phase in minutes
    """
    total_minutes = (time.time() - start_time) / 60.0
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    print(f"\n[{phase_name}] took {hours}h {minutes}min")
    return round(total_minutes, 3)


def _balanced_span(text, opening, closing, offset=None):
    """Locate the next matching delimiters outside quoted JSON strings."""
    start = text.find(opening, max(0, offset or 0))
    if start < 0:
        return None, None
    depth = 0
    quoted = escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == opening:
            depth += 1
        elif not quoted and char == closing:
            depth -= 1
            if depth == 0:
                return start, index
    return None, None


def _find_object_slice_bracket_scan(s, start_idx=None):
    return _balanced_span(s, "{", "}", start_idx)


def _find_array_slice_bracket_scan(s, start_idx=None):
    return _balanced_span(s, "[", "]", start_idx)


def _json_fragments(text, opening, closing):
    start, end = _balanced_span(text, opening, closing)
    if start is None:
        start, end = text.find(opening), text.rfind(closing)
        if start < 0 or end <= start:
            return
    while start is not None:
        yield text[start:end + 1]
        start, end = _balanced_span(text, opening, closing, end + 1)


def _repairs(fragment, *, allow_truncation):
    yield fragment
    sanitized = sanitize_json_like(fragment)
    if sanitized != fragment:
        yield sanitized
    if allow_truncation:
        shortened = _truncate_json_string_literals(sanitized)
        if shortened != sanitized:
            yield shortened


def _decode_fragment(fragment, *, array):
    for candidate in _repairs(fragment, allow_truncation=array):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if array:
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
        elif isinstance(value, dict):
            return value
    return None


def extract_json_object(text: str):
    """Prefer the first decodable object in the full response, with the JSON fence as fallback."""
    fence = _extract_json_fence_scope(text)
    scopes = (text, fence) if fence and fence != text else (text,)
    for scope in scopes:
        for fragment in _json_fragments(scope, "{", "}"):
            value = _decode_fragment(fragment, array=False)
            if value is not None:
                return value
    raise ValueError("No valid JSON object found in the text.")


def _escape_control_chars_in_json_strings(text: str) -> str:
    """
    Repair common LLM JSON mistakes where string values contain literal newlines,
    tabs, or carriage returns instead of escaped \\n / \\t / \\r sequences.
    """
    out = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == '\\':
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_str = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_str = True
    return "".join(out)


def sanitize_json_like(text: str) -> str:
    # Escape control characters inside JSON strings before json.loads sees them.
    text = _escape_control_chars_in_json_strings(text)
    # Escape backslashes that are not followed by a valid escape char: " \ / b f n r t u
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Skip complete string literals when removing trailing structural commas.
    tokens = r'"(?:\\.|[^"\\])*"|,\s*(?=\}|\])'
    return re.sub(tokens, lambda match: match[0] if match[0].startswith('"') else "", text, flags=re.DOTALL)


def _truncate_json_string_literals(text: str, max_len: int = 16000) -> str:
    """
    Keep JSON structure intact while shortening oversized string literals.
    This is a last-resort repair for responses that are parseable in structure
    except for extremely long evidence strings that were cut off or over-escaped.
    """
    if max_len <= 0:
        return text

    out = []
    in_str = False
    escape = False
    buf = []

    def _flush_buf():
        nonlocal buf
        if not buf:
            return
        content = "".join(buf)
        if len(content) > max_len:
            content = content[:max_len].rstrip() + " ..."
        out.append(content)
        buf = []

    for ch in text:
        if in_str:
            if escape:
                buf.append(ch)
                escape = False
                continue
            if ch == '\\':
                buf.append(ch)
                escape = True
                continue
            if ch == '"':
                _flush_buf()
                out.append('"')
                in_str = False
                continue
            buf.append(ch)
            continue

        out.append(ch)
        if ch == '"':
            in_str = True

    if in_str:
        _flush_buf()

    return "".join(out)


def _extract_json_fence_scope(text: str):
    """
    Return the inner content of a ```json fenced block, but DO NOT close the fence
    if the closing ``` appears inside a JSON string. This prevents early cutoff
    by lines like ```latex that live inside a JSON string value.
    Returns the 'scope' string or None if no json fence exists.
    """
    # Find the opening fence line: ^```[ \t]*json...
    open_pat = re.compile(r'^```[ \t]*json[^\n]*\n', re.IGNORECASE | re.MULTILINE)
    m = open_pat.search(text)
    if not m:
        return None

    i = m.end()                   # start scanning after opening fence newline
    n = len(text)
    in_str = False               # inside a JSON string? (double quotes only)
    escape = False               # previous char was a backslash
    line_start = i               # index of current line start

    while i < n:
        ch = text[i]

        if ch == '\n':
            # track line start
            line_start = i + 1

        if in_str:
            if escape:
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
        else:
            # not in a JSON string
            if ch == '"':
                in_str = True
            else:
                # Only consider a closing fence if it is at the start of a line
                # AND we're not in a JSON string.
                if i == line_start and text.startswith('```', i):
                    # Found the real closing fence
                    return text[m.end():i]

        i += 1

    # No closing fence found; treat until EOF as scope
    return text[m.end():]


def extract_json_array(text: str):
    """Read an array of records, preferring the content of an explicit JSON fence."""
    fence = _extract_json_fence_scope(text)
    scope = text if fence is None else fence
    found = False
    for fragment in _json_fragments(scope, "[", "]"):
        found = True
        value = _decode_fragment(fragment, array=True)
        if value is not None:
            return value
    if not found:
        raise ValueError("No JSON array found in the text.")
    raise ValueError("No valid JSON array of objects found.")
