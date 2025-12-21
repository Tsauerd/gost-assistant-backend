import re
from typing import List, Dict
from .topic_maps import TOPIC_RULES
from .validators import (
    validate_methods_dust_clay,
    validate_modulus_fineness,
    validate_cement_sieve_008,
    validate_frost_F200_agg
)

def _detect_topic(query: str) -> str | None:
    for rule in TOPIC_RULES:
        if any(p.search(query) for p in rule.query_patterns):
            return rule.name
    return None

def _check_sources(topic: str, used_standards: List[str]) -> Dict:
    rule = next((r for r in TOPIC_RULES if r.name == topic), None)
    if not rule:
        return {"ok": True, "reason": None}
    used = {str(s).lower() for s in used_standards}
    allow = {s.lower() for s in rule.allowed_gosts}
    forbid = {s.lower() for s in (rule.forbidden_gosts or [])}
    if used & forbid:
        return {"ok": False, "reason": f"Запрещённые источники: {', '.join(used & forbid)}"}
    if not (used & allow):
        return {"ok": False, "reason": f"Нет допустимых источников (ожидались: {', '.join(rule.allowed_gosts)})"}
    return {"ok": True, "reason": None}

def _validate(topic: str, answer_text: str) -> Dict:
    if topic == "dust_clay_in_coarse_agg":
        return validate_methods_dust_clay(answer_text)
    if topic == "modulus_fineness_sand":
        return validate_modulus_fineness(answer_text)
    if topic == "cement_fineness_sieve_008":
        return validate_cement_sieve_008(answer_text)
    if topic == "frost_agg_requirements_for_concrete":
        return validate_frost_F200_agg(answer_text)
    return {"ok": True}

def post_guard(user_query: str, answer_text: str, used_standards: List[str]) -> Dict:
    """
    Возвращает { ok: bool, answer: str, warnings: [..] }
    """
    topic = _detect_topic(user_query)
    warnings: List[str] = []

    if not topic:
        return {"ok": True, "answer": answer_text, "warnings": warnings}

    src = _check_sources(topic, used_standards)
    if not src["ok"]:
        warnings.append(src["reason"])
        answer_text = f"> Внимание: {src['reason']}\n\n" + answer_text

    val = _validate(topic, answer_text)
    if not val.get("ok", True):
        msg = "Недостаточная полнота: " + "; ".join(val.get("missing", []))
        warnings.append(msg)

        hints = {
            "dust_clay_in_coarse_agg": "\nЭталон: методы — отмучивание, пипеточный, мокрое просеивание; формулы П=(m−m1)/m·100 и П=(m−m2)/m·100.",
            "modulus_fineness_sand": "\nЭталон: сита 2.5; 1.25; 0.63; 0.315; 0.16 и Mк=(R2.5+R1.25+R0.63+R0.315+R0.16)/100.",
            "cement_fineness_sieve_008": "\nЭталон: 50 г; 5–7 мин; контроль вручную ≤ 0,05 г; сито №008.",
            "frost_agg_requirements_for_concrete": "\nЭталон: для заполнителя F200 — 200 циклов; потеря массы ≤5%; ссылаться на 8269.0-97/26633-2015, не на 10060."
        }
        answer_text += "\n\n---" + hints.get(topic, "")

    return {"ok": src["ok"] and val.get("ok", True), "answer": answer_text, "warnings": warnings}
