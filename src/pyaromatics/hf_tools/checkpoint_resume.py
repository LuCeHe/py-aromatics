"""Checkpoint continuation helpers (``loadckpt:``, ``[testckpt:…]``, ``[loadckpt:…]``)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

LOADCKPT_PREFIX = "loadckpt:"
INSTRUCT_FINETUNE_DATASET = "tulu3sft"
INSTRUCT_LR_SCALE = 0.01

_testckpt_BRACKET = re.compile(r"^\[testckpt:([^\]]+)\]\s*(.*)$", re.DOTALL)
_loadckpt_BRACKET = re.compile(r"^\[loadckpt:([^\]]+)\]\s*(.*)$", re.DOTALL)


@dataclass
class CheckpointSearchRoots:
    """Directories searched when resolving ``loadckpt:`` / ``[testckpt:…]`` ids."""

    exps_dir: str
    continue_checkpoints: str
    final_checkpoints: str
    extra_bases: tuple[str, ...] = field(default_factory=tuple)


def checkpoint_roots_from_paths_module(paths) -> CheckpointSearchRoots:
    """Build search roots from a ``paths`` module (``EXPSDIR``, ``CONTINUE_CHECKPOINTS``, …)."""
    extra = (os.path.join(paths.CONTINUE_CHECKPOINTS, "foldable_wiki103"),)
    return CheckpointSearchRoots(
        exps_dir=paths.EXPSDIR,
        continue_checkpoints=paths.CONTINUE_CHECKPOINTS,
        final_checkpoints=paths.FINAL_CHECKPOINTS,
        extra_bases=extra,
    )


def _strip_chained_loadckpt_prefixes(notes: str) -> str:
    s = (notes or "").strip()
    while True:
        s2 = s.lstrip()
        if s2.startswith(LOADCKPT_PREFIX):
            rest = s2[len(LOADCKPT_PREFIX) :].strip()
            parts = rest.split(None, 1)
            s = parts[1].strip() if len(parts) > 1 else ""
        elif s2.startswith("loadmodel:"):
            rest = s2[len("loadmodel:") :].strip()
            parts = rest.split(None, 1)
            s = parts[1].strip() if len(parts) > 1 else ""
        else:
            break
    return s


def parse_testckpt_bracket_notes(notes: str) -> Optional[Tuple[str, str]]:
    m = _testckpt_BRACKET.match((notes or "").strip())
    if not m:
        return None
    ckpt_id = os.path.basename(m.group(1).strip().rstrip("/\\"))
    tail = m.group(2).strip()
    return (ckpt_id, tail)


def parse_loadckpt_bracket_notes(notes: str) -> Optional[Tuple[str, str]]:
    m = _loadckpt_BRACKET.match((notes or "").strip())
    if not m:
        return None
    ckpt_id = os.path.basename(m.group(1).strip().rstrip("/\\"))
    tail = m.group(2).strip()
    return (ckpt_id, tail)


def notes_has_loadckpt(notes: str) -> bool:
    s = (notes or "").strip()
    if parse_loadckpt_bracket_notes(s):
        return True
    return s.startswith(LOADCKPT_PREFIX) or s.startswith("loadmodel:")


def notes_request_instruct_finetune(notes: str) -> bool:
    s = (notes or "").strip()
    bracket = parse_loadckpt_bracket_notes(s)
    if bracket and "_instruct" in bracket[1]:
        return True
    return "_instruct" in s


def _merge_args_txt_into_namespace(
    resolved_dir: Optional[str],
    args,
    preserve_keys: Tuple[str, ...] = ("output_dir", "epochs", "stop_time"),
) -> Optional[dict]:
    if not resolved_dir or not os.path.isdir(resolved_dir):
        return None
    path = os.path.join(resolved_dir, "args.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(saved, dict):
        return None
    preserved = {k: getattr(args, k, None) for k in preserve_keys}
    for key, val in saved.items():
        if key in preserve_keys:
            continue
        setattr(args, key, val)
    for key, val in preserved.items():
        setattr(args, key, val)
    return saved


def parse_loadckpt_checkpoint_id(notes: str) -> Optional[str]:
    s = (notes or "").strip()
    bracket = parse_loadckpt_bracket_notes(s)
    if bracket:
        return bracket[0]
    if s.startswith(LOADCKPT_PREFIX):
        rest = s.split(LOADCKPT_PREFIX, 1)[1].strip()
    elif s.startswith("loadmodel:"):
        rest = s.split("loadmodel:", 1)[1].strip()
    else:
        return None
    if not rest:
        return None
    first = rest.split(None, 1)[0]
    return os.path.basename(first.rstrip("/\\"))


def _resolve_loadckpt_dir(loaded_exp_id: str, roots: CheckpointSearchRoots) -> Optional[str]:
    name = os.path.basename(str(loaded_exp_id).strip().rstrip("/\\"))
    search_bases = (*roots.extra_bases, roots.continue_checkpoints, roots.final_checkpoints, roots.exps_dir)
    for base in search_bases:
        cand = os.path.join(base, name)
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "args.txt")):
            return cand
    return None


def _apply_loadckpt_from_checkpoint(
    args,
    ckpt_id: str,
    *,
    roots: CheckpointSearchRoots,
    continuation_tail: str = "",
) -> bool:
    resolved = _resolve_loadckpt_dir(ckpt_id, roots)
    if resolved is None:
        resolved = os.path.join(roots.exps_dir, ckpt_id)
    saved = _merge_args_txt_into_namespace(
        resolved, args, ("output_dir", "epochs", "stop_time"),
    )
    if saved is None:
        return False
    saved_notes = str(saved.get("notes", "")).strip()
    saved_core = _strip_chained_loadckpt_prefixes(saved_notes)
    merged_core = f"{saved_core}{continuation_tail}".strip() if continuation_tail else saved_core
    args.notes = (
        f"{LOADCKPT_PREFIX}{ckpt_id} {merged_core}".strip()
        if merged_core
        else f"{LOADCKPT_PREFIX}{ckpt_id}"
    )
    if notes_request_instruct_finetune(args.notes):
        args.dataset = INSTRUCT_FINETUNE_DATASET
        if "quicklueval" not in args.notes:
            args.notes = f"{args.notes}_quicklueval"
    return True


def apply_loadckpt_args_from_checkpoint(args, *, roots: CheckpointSearchRoots) -> None:
    ckpt_id = parse_loadckpt_checkpoint_id(getattr(args, "notes", "") or "")
    if not ckpt_id:
        return
    _apply_loadckpt_from_checkpoint(args, ckpt_id, roots=roots)


def is_resume_ckpt(args, *, roots: CheckpointSearchRoots) -> dict[str, Any]:
    """
    Merge checkpoint ``args.txt`` for ``loadckpt:`` / ``[loadckpt:…]`` / ``[testckpt:…]`` notes.

    Returns metadata: ``loadedckpt``, ``n_continuation``, ``continued_from_checkpoint``,
    ``loaded_exp_id``, ``resolved_load_dir``, and optionally ``testckpt_eval_only``.
    """
    bracket = parse_testckpt_bracket_notes(getattr(args, "notes", "") or "")
    if bracket:
        ckpt_id, eval_tail = bracket
        resolved_load_dir = _resolve_loadckpt_dir(ckpt_id, roots)
        if resolved_load_dir is None:
            resolved_load_dir = os.path.join(roots.exps_dir, ckpt_id)
        saved = _merge_args_txt_into_namespace(
            resolved_load_dir, args, ("output_dir", "epochs", "stop_time"),
        )
        saved_train_notes = ""
        if isinstance(saved, dict):
            saved_train_notes = str(saved.get("notes", "")).strip()
        eval_notes = eval_tail.strip()
        args.notes = eval_notes
        return {
            "loadedckpt": ckpt_id,
            "n_continuation": 0,
            "continued_from_checkpoint": None,
            "loaded_exp_id": ckpt_id,
            "resolved_load_dir": resolved_load_dir,
            "testckpt_eval_only": True,
            "testckpt_arch_notes": saved_train_notes or None,
        }

    load_bracket = parse_loadckpt_bracket_notes(getattr(args, "notes", "") or "")
    if load_bracket:
        ckpt_id, cont_tail = load_bracket
        _apply_loadckpt_from_checkpoint(args, ckpt_id, roots=roots, continuation_tail=cont_tail)
    else:
        apply_loadckpt_args_from_checkpoint(args, roots=roots)
    notes = args.notes or ""
    out: dict[str, Any] = {
        "loadedckpt": None,
        "n_continuation": 0,
        "continued_from_checkpoint": None,
        "loaded_exp_id": "None",
        "resolved_load_dir": None,
    }
    s = notes.strip()
    if not (s.startswith(LOADCKPT_PREFIX) or s.startswith("loadmodel:")):
        return out

    loaded_exp_id = parse_loadckpt_checkpoint_id(notes)
    if not loaded_exp_id:
        return out
    resolved_load_dir = _resolve_loadckpt_dir(loaded_exp_id, roots)
    if resolved_load_dir is None:
        resolved_load_dir = os.path.join(roots.exps_dir, loaded_exp_id)

    prev_results_path = os.path.join(resolved_load_dir, "results.txt")
    if os.path.isfile(prev_results_path):
        with open(prev_results_path, "r", encoding="utf-8") as f:
            prev_results = json.load(f)
        n_continuation = int(prev_results.get("n_continuation", 0)) + 1
        continued_from_checkpoint = os.path.basename(resolved_load_dir)
    else:
        n_continuation = 1
        continued_from_checkpoint = os.path.basename(resolved_load_dir)

    out["loadedckpt"] = loaded_exp_id
    out["n_continuation"] = n_continuation
    out["continued_from_checkpoint"] = continued_from_checkpoint
    out["loaded_exp_id"] = loaded_exp_id
    out["resolved_load_dir"] = resolved_load_dir
    return out
