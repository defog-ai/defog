"""Schema-aware repair for providers with loose JSON structured output.

The pipeline separates syntax recovery, deterministic schema fixes, field-level
model patches, and full-object fallback. Repair calls never receive the source
conversation.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Sequence

from pydantic import ValidationError

from .cost import CostCalculator


CompletionCall = Callable[[Dict[str, Any]], Awaitable[Any]]
UsageCalculator = Callable[[Any], tuple[int, int, Optional[int], Any]]


def deterministic_json_repair(content: str) -> str:
    """Apply conservative syntax fixes to common loose-JSON output."""
    value = content.strip()
    if value.startswith("```"):
        first_newline = value.find("\n")
        value = value[first_newline + 1 :] if first_newline != -1 else value[3:]
        if value.endswith("```"):
            value = value[:-3]
        value = value.strip()

    value = re.sub(r"(?m)^\s*//[^\n]*$", "", value)
    value = re.sub(r"//[^\n]*", "", value)
    value = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
    for source, replacement in {
        "True": "true",
        "False": "false",
        "None": "null",
        "NaN": "null",
        "undefined": "null",
        "Infinity": "null",
    }.items():
        value = re.sub(rf"\b{source}\b", replacement, value)
    value = re.sub(r"-\s*null\b", "null", value)
    value = re.sub(r",(\s*[}\]])", r"\1", value)

    if "'" in value and not re.search(r'(?<=[{\[,:])\s*"', value):
        candidate = value.replace("'", '"')
        try:
            json.loads(candidate)
            value = candidate
        except json.JSONDecodeError:
            pass

    open_braces = open_brackets = 0
    in_string = escape_next = False
    for character in value:
        if escape_next:
            escape_next = False
            continue
        if character == "\\" and in_string:
            escape_next = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            open_braces += 1
        elif character == "}":
            open_braces -= 1
        elif character == "[":
            open_brackets += 1
        elif character == "]":
            open_brackets -= 1

    if in_string:
        value += '"'
    tail = value.rstrip()
    if tail.endswith(":"):
        value = tail + " null"
    elif tail.endswith(","):
        value = tail[:-1]
    if open_brackets > 0:
        value += "]" * open_brackets
    if open_braces > 0:
        value += "}" * open_braces
    return re.sub(r",(\s*[}\]])", r"\1", value)


def _json_candidates(content: str) -> Iterable[str]:
    stripped = content.strip()
    yield stripped
    repaired = deterministic_json_repair(stripped)
    if repaired != stripped:
        yield repaired

    decoder = json.JSONDecoder()
    for index, character in enumerate(repaired):
        if character not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(repaired[index:])
        except json.JSONDecodeError:
            continue
        yield repaired[index : index + end]
        break


def _recover_json(content: str) -> tuple[Optional[Any], bool]:
    original = content.strip()
    seen = set()
    for candidate in _json_candidates(content):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate), candidate != original
        except (json.JSONDecodeError, TypeError):
            continue
    return None, False


def _json_pointer(location: Sequence[Any]) -> str:
    escaped = (str(part).replace("~", "~0").replace("/", "~1") for part in location)
    return "/" + "/".join(escaped)


def _get_at_path(value: Any, location: Sequence[Any]) -> Any:
    current = value
    for part in location:
        current = current[part]
    return current


def _set_at_path(value: Any, location: Sequence[Any], replacement: Any) -> bool:
    if not location:
        return False
    try:
        parent = _get_at_path(value, location[:-1])
        part = location[-1]
        if isinstance(parent, dict):
            parent[part] = replacement
        elif (
            isinstance(parent, list)
            and isinstance(part, int)
            and 0 <= part < len(parent)
        ):
            parent[part] = replacement
        else:
            return False
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _delete_at_path(value: Any, location: Sequence[Any]) -> bool:
    if not location:
        return False
    try:
        parent = _get_at_path(value, location[:-1])
        part = location[-1]
        if isinstance(parent, dict) and part in parent:
            del parent[part]
            return True
        if (
            isinstance(parent, list)
            and isinstance(part, int)
            and 0 <= part < len(parent)
        ):
            del parent[part]
            return True
    except (KeyError, IndexError, TypeError):
        pass
    return False


def _validation_errors(error: ValidationError) -> list[Dict[str, Any]]:
    return error.errors(include_url=False, include_context=False, include_input=True)


def _apply_deterministic_schema_repairs(
    candidate: Any, errors: Sequence[Dict[str, Any]]
) -> int:
    repaired = 0
    for error in errors:
        location = tuple(error.get("loc", ()))
        if (
            error.get("type") == "string_type"
            and error.get("input") is None
            and _set_at_path(candidate, location, "")
        ):
            repaired += 1
        elif error.get("type") == "extra_forbidden" and _delete_at_path(
            candidate, location
        ):
            repaired += 1
    return repaired


def _resolve_ref(schema: Any, root_schema: Dict[str, Any]) -> Any:
    seen = set()
    while isinstance(schema, dict) and "$ref" in schema:
        reference = schema["$ref"]
        if reference in seen or not reference.startswith("#/"):
            break
        seen.add(reference)
        current: Any = root_schema
        try:
            for part in reference[2:].split("/"):
                current = current[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError):
            break
        schema = current
    return schema


def _schema_for_location(
    root_schema: Dict[str, Any], location: Sequence[Any]
) -> Dict[str, Any]:
    current: Any = root_schema
    for part in location:
        current = _resolve_ref(current, root_schema)
        if not isinstance(current, dict):
            return {}
        variants = current.get("anyOf") or current.get("oneOf")
        if variants:
            useful = [
                _resolve_ref(option, root_schema)
                for option in variants
                if isinstance(option, dict) and option.get("type") != "null"
            ]
            if useful:
                current = useful[0]
        if isinstance(part, int):
            current = current.get("items", {})
        else:
            properties = current.get("properties", {})
            if part not in properties:
                return {
                    "additionalProperties": current.get("additionalProperties", True)
                }
            current = properties[part]
    resolved = _resolve_ref(current, root_schema)
    return copy.deepcopy(resolved) if isinstance(resolved, dict) else {}


def _small_parent_context(candidate: Any, location: Sequence[Any]) -> Dict[str, Any]:
    parent_location = tuple(location[:-1])
    try:
        parent = _get_at_path(candidate, parent_location)
    except (KeyError, IndexError, TypeError):
        parent = None
    serialized = json.dumps(parent, ensure_ascii=False, default=str)
    if len(serialized) > 6000 and location:
        try:
            current_value = _get_at_path(candidate, location)
        except (KeyError, IndexError, TypeError):
            current_value = None
        parent = {str(location[-1]): current_value}
    return {
        "parent_path": _json_pointer(parent_location) if parent_location else "",
        "value": parent,
    }


def _base_metadata(strategy: str, syntax_repaired: bool) -> Dict[str, Any]:
    return {
        "strategy": strategy,
        "attempts": 0,
        "deterministic_fields": 0,
        "model_patched_fields": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "cost_in_cents": 0.0,
        "syntax_repaired": syntax_repaired,
        "success": False,
    }


def _record_usage(
    metadata: Dict[str, Any],
    response: Any,
    model: str,
    usage_calculator: UsageCalculator,
) -> None:
    input_tokens, output_tokens, cached_tokens, _ = usage_calculator(response)
    cached_tokens = cached_tokens or 0
    metadata["input_tokens"] += input_tokens
    metadata["output_tokens"] += output_tokens
    metadata["cached_input_tokens"] += cached_tokens

    usage = getattr(response, "usage", None)
    reported_cost = getattr(usage, "cost", None) if usage else None
    if reported_cost is not None:
        repair_cost = reported_cost * 100
        metadata["cost_source"] = "provider"
    else:
        repair_cost = CostCalculator.calculate_cost(
            model, input_tokens, output_tokens, cached_tokens
        )
        if repair_cost is not None:
            metadata.setdefault("cost_source", "local")
    if repair_cost is None:
        metadata["cost_in_cents"] = None
    elif metadata["cost_in_cents"] is not None:
        metadata["cost_in_cents"] += repair_cost


def _completion_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("repair call returned no choices")
    return getattr(choices[0].message, "content", None) or ""


def _repair_request(
    *,
    model: str,
    payload: Dict[str, Any],
    system_message: str,
    max_tokens: int,
    request_options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if request_options:
        params.update(request_options)
    return params


async def repair_structured_response(
    *,
    raw_content: str,
    response_format: Any,
    model: str,
    create_completion: CompletionCall,
    usage_calculator: UsageCalculator,
    metadata: Optional[Dict[str, Any]] = None,
    request_options: Optional[Dict[str, Any]] = None,
) -> Any:
    """Parse and repair a structured response with at most one model call."""
    if not response_format or not raw_content:
        return raw_content

    parsed, syntax_repaired = _recover_json(raw_content)
    has_pydantic = hasattr(response_format, "model_validate")
    if parsed is None:
        repair_metadata = _base_metadata("full_object", False)
        repair_metadata["attempts"] = 1
        schema = (
            response_format.model_json_schema()
            if hasattr(response_format, "model_json_schema")
            else {}
        )
        params = _repair_request(
            model=model,
            payload={"broken_output": raw_content, "expected_schema": schema},
            system_message=(
                "Repair the broken output into one valid JSON object matching the "
                "provided schema. Return JSON only. The source conversation is "
                "intentionally unavailable; do not invent missing source facts."
            ),
            max_tokens=min(16384, max(1024, len(raw_content.encode()) // 3 + 512)),
            request_options=request_options,
        )
        try:
            response = await create_completion(params)
            _record_usage(repair_metadata, response, model, usage_calculator)
            repaired, _ = _recover_json(_completion_content(response))
            if repaired is None:
                raise ValueError("full-object repair returned invalid JSON")
            result = (
                response_format.model_validate(repaired) if has_pydantic else repaired
            )
            repair_metadata["success"] = True
            if metadata is not None:
                metadata.update(repair_metadata)
            return result
        except Exception:
            if metadata is not None:
                metadata.update(repair_metadata)
            return raw_content

    if not has_pydantic:
        if syntax_repaired and metadata is not None:
            repair_metadata = _base_metadata("deterministic", True)
            repair_metadata["success"] = True
            metadata.update(repair_metadata)
        return parsed

    try:
        result = response_format.model_validate(parsed)
        if syntax_repaired and metadata is not None:
            repair_metadata = _base_metadata("deterministic", True)
            repair_metadata["success"] = True
            metadata.update(repair_metadata)
        return result
    except ValidationError as initial_error:
        candidate = copy.deepcopy(parsed)
        repair_metadata = _base_metadata("field_patch", syntax_repaired)
        repair_metadata["deterministic_fields"] = _apply_deterministic_schema_repairs(
            candidate, _validation_errors(initial_error)
        )

    try:
        result = response_format.model_validate(candidate)
        repair_metadata["strategy"] = "deterministic"
        repair_metadata["success"] = True
        if metadata is not None:
            metadata.update(repair_metadata)
        return result
    except ValidationError as unresolved_error:
        unresolved = _validation_errors(unresolved_error)

    locations: Dict[str, tuple[Any, ...]] = {}
    for error in unresolved:
        location = tuple(error.get("loc", ()))
        if not location:
            if metadata is not None:
                metadata.update(repair_metadata)
            return raw_content
        locations.setdefault(_json_pointer(location), location)

    schema = response_format.model_json_schema()
    invalid_fields = [
        {
            "path": _json_pointer(tuple(error["loc"])),
            "error": {"type": error.get("type", ""), "message": error.get("msg", "")},
            "field_schema": _schema_for_location(schema, tuple(error["loc"])),
            "context": _small_parent_context(candidate, tuple(error["loc"])),
        }
        for error in unresolved
    ]
    params = _repair_request(
        model=model,
        payload={
            "invalid_fields": invalid_fields,
            "response_contract": {
                "repairs": [
                    {
                        "path": "one exact invalid_fields path",
                        "value": "replacement JSON value",
                    }
                ]
            },
        },
        system_message=(
            "Patch only the listed invalid JSON fields. Return one JSON object "
            'with a "repairs" array containing exactly one entry per unique '
            'invalid path. Each entry must have "path" and the replacement '
            '"value". Never return or alter an unlisted path.'
        ),
        max_tokens=min(4096, max(256, len(locations) * 192 + 128)),
        request_options=request_options,
    )
    repair_metadata["attempts"] = 1
    try:
        response = await create_completion(params)
        _record_usage(repair_metadata, response, model, usage_calculator)
        patch_object, _ = _recover_json(_completion_content(response))
        repairs = (
            patch_object.get("repairs") if isinstance(patch_object, dict) else None
        )
        if not isinstance(repairs, list):
            raise ValueError("field repair did not return a repairs array")

        received_paths = []
        for repair in repairs:
            if (
                not isinstance(repair, dict)
                or "path" not in repair
                or "value" not in repair
            ):
                raise ValueError("invalid repair entry")
            path = repair["path"]
            if path not in locations:
                raise ValueError("repair returned a path outside the allow-list")
            if path in received_paths:
                raise ValueError("repair returned a duplicate path")
            received_paths.append(path)
        if set(received_paths) != set(locations):
            raise ValueError("repair did not cover every invalid path")

        patched = copy.deepcopy(candidate)
        for repair in repairs:
            if not _set_at_path(patched, locations[repair["path"]], repair["value"]):
                raise ValueError("repair path could not be applied")
        result = response_format.model_validate(patched)
        repair_metadata["model_patched_fields"] = len(repairs)
        repair_metadata["success"] = True
        if metadata is not None:
            metadata.update(repair_metadata)
        return result
    except Exception:
        if metadata is not None:
            metadata.update(repair_metadata)
        return raw_content
