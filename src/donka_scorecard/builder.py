from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from .metrics import binary_metrics, numeric_population_stability, population_stability, woe_iv_by_bin


def _read_config(path: Path) -> tuple[dict[str, Any], Path]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("config version must be 1")
    for key in ("data", "target", "features", "windows", "pack"):
        if key not in config:
            raise ValueError(f"config is missing {key!r}")
    config["features"] = sorted(config["features"], key=lambda feature: feature["name"])
    names = [feature["name"] for feature in config["features"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("features must have unique names")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in names):
        raise ValueError("feature names must be lowercase identifiers")
    if config["target"].get("bad_label") == config["target"].get("good_label"):
        raise ValueError("target good_label and bad_label must differ")
    return config, path.parent


def _date_mask(values: pd.Series, start: str, end: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return (parsed >= pd.Timestamp(start, tz="UTC")) & (parsed < pd.Timestamp(end, tz="UTC"))


def _metric_population(frame: pd.DataFrame, target: pd.Series, features: list[dict[str, Any]], conf: dict[str, Any]):
    columns = [feature["column"] for feature in features]
    subset = frame.loc[:, columns].copy()
    for feature in features:
        if feature["type"] == "categorical":
            subset[feature["column"]] = subset[feature["column"]].astype("object")
    return subset, target.astype(int).to_numpy()


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("scorecard produced a non-finite points value")
    return number


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _special_markers(features: list[dict[str, Any]], frame: pd.DataFrame) -> tuple[dict[str, dict[Any, Any]], list[Any]]:
    """Give feature-local special values unique markers for OptBinning's shared-code API."""
    mappings: dict[str, dict[Any, Any]] = {}
    markers: list[Any] = []
    numeric_columns = [feature["column"] for feature in features if feature["type"] == "numeric"]
    numeric_values = pd.concat([pd.to_numeric(frame[column], errors="coerce") for column in numeric_columns]) if numeric_columns else pd.Series(dtype=float)
    current = float(numeric_values.min()) if numeric_values.notna().any() else 0.0
    for feature in features:
        specials = feature.get("special_values", [])
        if not specials:
            continue
        mapping = {}
        if feature["type"] == "numeric":
            for special in specials:
                current = float(np.nextafter(current, -np.inf))
                if not math.isfinite(current):
                    raise ValueError(f"cannot allocate a finite special-value marker for {feature['name']}")
                mapping[special] = current
                markers.append(current)
        else:
            present = set(frame[feature["column"]].dropna().astype(str))
            index = 0
            for special in specials:
                while True:
                    marker = f"__DONKA_SPECIAL_{index}__"
                    index += 1
                    if marker not in present:
                        break
                mapping[special] = marker
                markers.append(marker)
        mappings[feature["column"]] = mapping
    return mappings, markers


def _prepare_features(frame: pd.DataFrame, features: list[dict[str, Any]], mappings: dict[str, dict[Any, Any]]) -> pd.DataFrame:
    prepared = frame[[feature["column"] for feature in features]].copy()
    for column, mapping in mappings.items():
        prepared[column] = prepared[column].replace(mapping)
    return prepared


def _safe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _labels(text: str) -> dict[str, str]:
    return {"en": text, "fr": text}


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        root = root.setdefault(part, {})
    root[parts[-1]] = value


def _get_path(root: dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        if not isinstance(root, dict) or part not in root:
            return None
        root = root[part]
    return root


def _matches(selector: str, value: Any) -> bool:
    selector = selector.strip()
    if not selector:
        return True
    if selector == "null":
        return value is None or (isinstance(value, float) and math.isnan(value))
    if value is None:
        return False
    quoted = re.findall(r'"(?:\\.|[^"\\])*"', selector)
    if quoted:
        return value in [json.loads(item) for item in quoted]
    try:
        number = float(value)
        if selector.startswith("< "):
            return number < float(selector[2:])
        if selector.startswith(">= "):
            return number >= float(selector[3:])
        interval = re.fullmatch(r"\[(-?\d+(?:\.\d+)?)\.\.(-?\d+(?:\.\d+)?)\)", selector)
        if interval:
            low, high = interval.groups()
            return (not low or number >= float(low)) and (not high or number < float(high))
        if re.fullmatch(r"-?\d+(?:\.\d+)?", selector):
            return number == float(selector)
    except (TypeError, ValueError):
        return False
    return False


def _evaluate_graph(graph: dict[str, Any], sample_input: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    points: dict[str, float] = {}
    for feature in features:
        node = nodes[f"feature_{feature['name']}"]
        content = node["content"]
        value = _get_path(sample_input, content["inputs"][0]["field"])
        for rule in content["rules"]:
            if _matches(rule[f"{feature['name']}_i"], value):
                points[feature["name"]] = float(rule[f"{feature['name']}_o"])
                break
        else:
            raise ValueError(f"generated graph has no matching bin for {feature['name']}={value!r}")
    score_expression = nodes["score"]["content"]["expressions"][0]["value"]
    base = float(score_expression.split(" + ", 1)[0])
    score = base + sum(points.values())
    band_node = nodes["band"]["content"]
    for rule in band_node["rules"]:
        if _matches(rule["band_i"], score):
            return {"score": score, "band": json.loads(rule["band_o"]), "decision": json.loads(rule["decision_o"])}
    raise ValueError("generated graph has no matching score band")


def _input_schema(features: list[dict[str, Any]]) -> dict[str, Any]:
    root: dict[str, Any] = {"type": "object", "required": [], "properties": {}}
    for feature in features:
        target = root
        parts = feature["field"].split(".")
        for part in parts[:-1]:
            props = target.setdefault("properties", {})
            child = props.setdefault(part, {"type": "object", "required": [], "properties": {}})
            if part not in target.setdefault("required", []):
                target["required"].append(part)
            target = child
        name = parts[-1]
        props = target.setdefault("properties", {})
        if feature["type"] == "categorical":
            values = list(feature["values"])
            values.extend(value for value in feature.get("special_values", []) if value not in values)
            field = {"type": ["string", "null"], "enum": values + ([None] if None not in values else [])}
        else:
            number_schema: dict[str, Any] = {"type": "number"}
            if "minimum" in feature:
                number_schema["minimum"] = feature["minimum"]
            if "maximum" in feature:
                number_schema["maximum"] = feature["maximum"]
            special_values = list(feature.get("special_values", []))
            if special_values:
                field = {"anyOf": [number_schema, {"enum": special_values}, {"type": "null"}]}
            else:
                field = {**number_schema, "type": ["number", "null"]}
        field["x-donka"] = {"label": _labels(feature.get("label", feature["name"]))}
        props[name] = field
        if name not in target.setdefault("required", []):
            target["required"].append(name)
    return {"$schema": "http://json-schema.org/draft-07/schema#", **root}


def _bin_expression(feature: dict[str, Any], binning: Any, bin_id: int) -> str:
    """Return a JDM unary test for an OptBinning bin index."""
    splits = list(getattr(binning, "splits", []))
    regular_bins = len(splits) + (1 if feature["type"] == "numeric" else 0)
    if bin_id == regular_bins + 1:
        return "null"
    if bin_id == regular_bins:
        special_values = feature.get("special_values", [])
        if not special_values:
            # No value can reach this reserved empty special bin.
            return "null"
        return ", ".join(json.dumps(value) for value in special_values)
    if feature["type"] == "numeric":
        if bin_id == 0:
            return f"< {float(splits[0]):.12g}"
        if bin_id == len(splits):
            return f">= {float(splits[-1]):.12g}"
        low, high = float(splits[bin_id - 1]), float(splits[bin_id])
        return f"[{low:.12g}..{high:.12g})"
    categories = getattr(binning, "splits", [])[bin_id]
    if isinstance(categories, (str, bytes)):
        categories = [categories]
    return ", ".join(json.dumps(v) for v in list(categories))


def _make_graph(config: dict[str, Any], scorecard: Scorecard) -> dict[str, Any]:
    table = scorecard.table(style="detailed")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    request_id = "request"
    nodes.append({
        "id": request_id,
        "type": "inputNode",
        "name": "Scorecard input",
        "content": {"schema": json.dumps(_input_schema(config["features"]), ensure_ascii=False, indent=2)},
        "position": {"x": 0, "y": 0},
    })
    previous = request_id
    expression_terms: list[str] = []
    for index, feature in enumerate(config["features"]):
        key = feature["name"]
        point_rows = table[table["Variable"] == feature["column"]]
        binning = scorecard.binning_process_.get_binned_variable(feature["column"])
        rules = []
        rows = point_rows.to_dict(orient="records")
        rows.sort(key=lambda row: (0 if row["Bin"] == "Special" else 2 if row["Bin"] == "Missing" else 1, row["Bin id"]))
        for row in rows:
            if str(row["Bin"]) == "Special" and not feature.get("special_values"):
                continue
            selector = _bin_expression(feature, binning, int(row["Bin id"]))
            output = format(_finite(row["Points"]), ".12g")
            rules.append({"_id": f"{key}_r{len(rules)}", f"{key}_i": selector, f"{key}_o": output})
        if not rules:
            raise ValueError(f"no scorecard bins were produced for feature {key}")
        node_id = f"feature_{key}"
        nodes.append({
            "id": node_id,
            "type": "decisionTableNode",
            "name": f"Points: {feature.get('label', key)}",
            "content": {
                "hitPolicy": "first", "passThrough": True,
                "inputs": [{"id": f"{key}_i", "name": feature.get("label", key), "field": feature["field"]}],
                "outputs": [{"id": f"{key}_o", "name": "Points", "field": f"points.{key}"}],
                "rules": rules,
            },
            "position": {"x": 320 + index % 3 * 320, "y": 180 + index // 3 * 220},
        })
        edges.append({"id": f"edge_{len(edges)}", "type": "edge", "sourceId": previous, "targetId": node_id})
        previous = node_id
        expression_terms.append(f"points.{key}")
    base_value = format(_finite(scorecard.intercept_), ".12g")
    score_value = " + ".join([base_value, *expression_terms])
    nodes.append({
        "id": "score", "type": "expressionNode", "name": "Score",
        "content": {"expressions": [{"id": "score_value", "key": "score", "value": score_value}], "passThrough": True},
        "position": {"x": 320, "y": 700},
    })
    edges.append({"id": f"edge_{len(edges)}", "type": "edge", "sourceId": previous, "targetId": "score"})
    bands = config["pack"]["bands"]
    band_rules = []
    for index, band in enumerate(bands):
        if index == 0:
            selector = f"< {band['max']}"
        elif index == len(bands) - 1:
            selector = ""
        else:
            selector = f"[{bands[index - 1]['max']}..{band['max']})"
        band_rules.append({"_id": f"band_r{index}", "band_i": selector, "band_o": json.dumps(band["name"]), "decision_o": json.dumps(band["decision"])})
    nodes.append({
        "id": "band", "type": "decisionTableNode", "name": "Score band",
        "content": {"hitPolicy": "first", "passThrough": True,
            "inputs": [{"id": "band_i", "name": "Score", "field": "score"}],
            "outputs": [{"id": "band_o", "name": "Band", "field": "band"}, {"id": "decision_o", "name": "Decision", "field": "decision"}],
            "rules": band_rules},
        "position": {"x": 640, "y": 700},
    })
    edges.append({"id": f"edge_{len(edges)}", "type": "edge", "sourceId": "score", "targetId": "band"})
    nodes.extend([
        {"id": "result", "type": "expressionNode", "name": "Scorecard result",
         "content": {"expressions": [
             {"id": "out_score", "key": "score", "value": "score"},
             {"id": "out_band", "key": "band", "value": "band"},
             {"id": "out_decision", "key": "decision", "value": "decision"},
         ], "passThrough": False}, "position": {"x": 960, "y": 700}},
        {"id": "output", "type": "outputNode", "name": "Scorecard result", "position": {"x": 1280, "y": 700}},
    ])
    edges.append({"id": f"edge_{len(edges)}", "type": "edge", "sourceId": "band", "targetId": "result"})
    edges.append({"id": f"edge_{len(edges)}", "type": "edge", "sourceId": "result", "targetId": "output"})
    return {"nodes": nodes, "edges": edges}


def build(config_path: Path) -> Path:
    """Train a configured scorecard and write a deterministic Donka pack and report."""
    config_path = config_path.resolve()
    config, base = _read_config(config_path)
    data_path = (base / config["data"]).resolve()
    if not data_path.is_file():
        raise ValueError(f"training data not found: {data_path}")
    frame = pd.read_csv(data_path)
    date_column = config["windows"]["date_column"]
    target_column = config["target"]["column"]
    required_columns = {date_column, target_column, *(feature["column"] for feature in config["features"])}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"training data is missing columns: {', '.join(missing_columns)}")
    for feature in config["features"]:
        feature.setdefault("field", feature["name"])
        if feature["type"] not in {"numeric", "categorical"}:
            raise ValueError(f"unsupported type for {feature['name']}: {feature['type']}")
    bad_label = config["target"]["bad_label"]
    good_label = config["target"]["good_label"]
    if not set(frame[target_column].dropna().unique()).issubset({bad_label, good_label}):
        raise ValueError("target column contains values outside good_label and bad_label")
    frame = frame[frame[target_column].isin([bad_label, good_label])].copy()
    windows = config["windows"]
    development_mask = _date_mask(frame[date_column], windows["development_start"], windows["development_end"])
    oot_mask = _date_mask(frame[date_column], windows["out_of_time_start"], windows["out_of_time_end"])
    development = frame.loc[development_mask].copy()
    oot = frame.loc[oot_mask].copy()
    minimum_bad = int(config["target"].get("minimum_bad_outcomes", 20))
    bad_count = int((development[target_column] == bad_label).sum())
    if bad_count < minimum_bad:
        raise ValueError(f"too few bad outcomes: found {bad_count}, need at least {minimum_bad} in development window")
    if len(oot) == 0:
        raise ValueError("out-of-time window contains no labeled rows")
    columns = [feature["column"] for feature in config["features"]]
    x_train = development[columns].copy()
    x_oot = oot[columns].copy()
    types = {feature["column"]: ("numerical" if feature["type"] == "numeric" else "categorical") for feature in config["features"]}
    special_mappings, special_markers = _special_markers(config["features"], frame)
    binning_fit_params = {}
    for feature in config["features"]:
        params = {}
        if feature.get("monotonic_trend"):
            params["monotonic_trend"] = feature["monotonic_trend"]
        if params:
            binning_fit_params[feature["column"]] = params
    binning = BinningProcess(variable_names=columns, categorical_variables=[c for c in columns if types[c] == "categorical"],
        min_bin_size=float(config.get("binning", {}).get("minimum_bin_fraction", 0.05)),
        max_n_bins=int(config.get("binning", {}).get("maximum_bins", 5)),
        min_n_bins=int(config.get("binning", {}).get("minimum_bins", 2)),
        max_n_prebins=int(config.get("binning", {}).get("prebins", 20)),
        min_prebin_size=float(config.get("binning", {}).get("minimum_prebin_fraction", 0.02)),
        special_codes=special_markers or None, binning_fit_params=binning_fit_params or None)
    x_train = _prepare_features(x_train, config["features"], special_mappings)
    x_oot = _prepare_features(x_oot, config["features"], special_mappings)
    model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=0)
    scorecard = Scorecard(
        binning_process=binning, estimator=model, scaling_method="pdo_odds",
        scaling_method_params={"pdo": config["pack"]["points_to_double_odds"], "odds": config["pack"]["base_odds"], "scorecard_points": config["pack"]["base_points"]},
        intercept_based=True, reverse_scorecard=bool(config["pack"].get("higher_score_is_safer", True)), rounding=True)
    y_train = (development[target_column] == bad_label).astype(int).to_numpy()
    scorecard.fit(x_train, y_train)
    probs = scorecard.estimator_.predict_proba(scorecard.binning_process_.transform(x_oot))[:, 1]
    metrics = binary_metrics((oot[target_column] == bad_label).astype(int).to_numpy(), probs)
    scored_oot = scorecard.score(x_oot)
    score_edges = np.quantile(scored_oot, [0.2, 0.5, 0.8]).tolist()
    bands = config["pack"].setdefault("bands", [
        {"name": "Review", "max": round(score_edges[0], 2), "decision": "review"},
        {"name": "Standard", "max": round(score_edges[1], 2), "decision": "standard"},
        {"name": "Preferred", "max": round(score_edges[2], 2), "decision": "preferred"},
        {"name": "Prime", "decision": "prime"},
    ])
    pack_key = config["pack"]["key"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", pack_key):
        raise ValueError("pack.key must contain lowercase letters, digits and hyphens")
    output = (base / config.get("output", "dist")).resolve() / pack_key
    output.mkdir(parents=True, exist_ok=True)
    (output / "decisions").mkdir(exist_ok=True)
    graph = _make_graph(config, scorecard)
    (output / "decisions" / "scorecard.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": 1, "key": pack_key,
        "name": {"en": config["pack"]["name_en"], "fr": config["pack"]["name_fr"]},
        "description": {"en": config["pack"]["description_en"], "fr": config["pack"]["description_fr"]},
        "version": config["pack"].get("version", "1.0"),
        "market": config["pack"].get("market", "CM"), "currency": config["pack"].get("currency", "XAF"),
        "entry": "scorecard", "decisions": ["scorecard"],
        "decisionLog": {"outcomeField": "decision", "redactedFields": []},
    }
    (output / "pack.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scenario_count = int(config.get("pack", {}).get("scenario_count", 8))
    selected_indices: list[Any] = []
    for feature in config["features"]:
        for special in feature.get("special_values", []):
            matching = oot.index[oot[feature["column"]] == special]
            if len(matching) and matching[0] not in selected_indices:
                selected_indices.append(matching[0])
        missing = oot.index[oot[feature["column"]].isna()]
        if len(missing) and missing[0] not in selected_indices:
            selected_indices.append(missing[0])
    for index in oot.index:
        if len(selected_indices) >= scenario_count:
            break
        if index not in selected_indices:
            selected_indices.append(index)
    selected_indices = selected_indices[:scenario_count]
    sample_rows = oot.loc[selected_indices]
    scenarios = []
    for row_idx, (_, row) in enumerate(sample_rows.iterrows(), start=1):
        row_frame = _prepare_features(pd.DataFrame([{column: row[column] for column in columns}]), config["features"], special_mappings)
        raw_score = _finite(scorecard.score(row_frame)[0])
        selected = bands[-1]
        for band in bands[:-1]:
            if raw_score < float(band["max"]):
                selected = band
                break
        decision = selected["decision"]
        band_name = selected["name"]
        sample_input: dict[str, Any] = {}
        for feature in config["features"]:
            value = row[feature["column"]]
            if pd.isna(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            _set_path(sample_input, feature["field"], value)
        graph_result = _evaluate_graph(graph, sample_input, config["features"])
        if not math.isclose(graph_result["score"], raw_score, abs_tol=0.01):
            raise ValueError(f"generated graph score does not match Python for scenario {row_idx}: {graph_result['score']} != {raw_score}")
        if graph_result["band"] != band_name or graph_result["decision"] != decision:
            raise ValueError(f"generated graph band does not match Python for scenario {row_idx}")
        scenarios.append({"decision": "scorecard", "name": f"Out-of-time sample {row_idx}", "input": sample_input,
            "expected": {"score": round(raw_score, 2), "band": band_name, "decision": decision}, "match": "exact"})
    (output / "scenarios.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "guide.md").write_text(config["pack"]["guide_en"].rstrip() + "\n", encoding="utf-8")
    (output / "guide.fr.md").write_text(config["pack"]["guide_fr"].rstrip() + "\n", encoding="utf-8")
    development_bins = scorecard.binning_process_.transform(x_train, metric="indices")
    oot_bins = scorecard.binning_process_.transform(x_oot, metric="indices")
    report = {
        "target": {"column": target_column, "bad_label": bad_label, "good_label": good_label,
            "definition": config["target"].get("definition", "")},
        "windows": windows,
        "sample_sizes": {"development": len(development), "development_bad": bad_count, "out_of_time": len(oot),
            "out_of_time_bad": int((oot[target_column] == bad_label).sum())},
        "out_of_time_metrics": metrics,
        "population_stability": {"score": numeric_population_stability(scorecard.score(x_train), scored_oot),
            "features": {feature["name"]: population_stability(development_bins[feature["column"]], oot_bins[feature["column"]]) for feature in config["features"]}},
        "features": [],
        "known_limits": ["Accepted-only samples create selection bias; declined applicants have no observed repayment outcome.",
            "Small samples make bins and performance estimates unstable; validate with a larger, representative population."],
        "source_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    detail = scorecard.table(style="detailed")
    for feature in config["features"]:
        rows = detail[detail["Variable"] == feature["column"]]
        independent = woe_iv_by_bin(y_train, development_bins[feature["column"]].to_numpy())
        independent_by_id = {row["bin"]: row for row in independent}
        bin_rows = rows.replace({np.nan: None}).to_dict(orient="records")
        for bin_row in bin_rows:
            check = independent_by_id.get(int(bin_row["Bin id"]))
            if check:
                bin_row["independent_woe"] = round(check["woe"], 8)
                bin_row["independent_iv"] = round(check["iv"], 8)
        report["features"].append({"name": feature["name"], "type": feature["type"], "bins": bin_rows,
            "independent_iv_total": round(sum(item["iv"] for item in independent), 8)})
    (base / (config.get("report", "model-report.json"))).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return output
