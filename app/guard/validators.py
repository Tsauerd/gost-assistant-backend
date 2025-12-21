import re
from typing import Dict, Tuple

def _all(text: str, pats: Tuple[str, ...]) -> bool:
    return all(re.search(p, text, re.I) for p in pats)

def _any(text: str, pats: Tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.I) for p in pats)

def validate_methods_dust_clay(answer: str) -> Dict:
    must = (r"отмуч", r"пипеточ", r"мокро(е|го)\s+просеиван")
    formulas = (
        r"П\s*=\s*\(\s*m\s*-\s*m1?\s*\)\s*/\s*m\s*\*\s*100",
        r"П\s*=\s*\(\s*m\s*-\s*m2?\s*\)\s*/\s*m\s*\*\s*100",
    )
    return {
        "ok": _all(answer, must) and _all(answer, formulas),
        "missing": [ "Методы: отмучивание/пипеточный/мокрое просеивание" ][:_all(answer, must) is False],
        "formulas": _all(answer, formulas)
    }

def validate_modulus_fineness(answer: str) -> Dict:
    sieves = (r"2[,\.]?\s*5", r"1[,\.]?\s*25", r"0[,\.]?\s*63", r"0[,\.]?\s*315", r"0[,\.]?\s*16")
    formula = r"M[кf]\s*=\s*\(?\s*R.*2\s*5.*\+\s*R.*1\s*25.*\+\s*R.*0\s*63.*\+\s*R.*0\s*315.*\+\s*R.*0\s*16.*\)\s*/\s*100"
    return {
        "ok": _all(answer, sieves) and re.search(formula, answer, re.I) is not None,
        "missing": [] if _all(answer, sieves) else ["Сита: 2.5; 1.25; 0.63; 0.315; 0.16"],
        "formula": re.search(formula, answer, re.I) is not None
    }

def validate_cement_sieve_008(answer: str) -> Dict:
    need = (r"50\s*г", r"5\s*[-–]\s*7\s*мин", r"0[,\.]?05\s*г", r"сито\s*№\s*0?08")
    return {"ok": _all(answer, need), "missing": [p for p in need if re.search(p, answer, re.I) is None]}

def validate_frost_F200_agg(answer: str) -> Dict:
    cycles = r"(200\s*цикл[ао]|числ[оа]\s*циклов.*200)"
    massloss = r"(потер[яи]\s*массы.*(≤|не\s*более)\s*5\s*%|5\s*%)"
    wrong_10060 = r"ГОСТ\s*10060"
    ok = re.search(cycles, answer, re.I) and re.search(massloss, answer, re.I) and not re.search(wrong_10060, answer, re.I)
    return {
        "ok": bool(ok),
        "missing": [] if ok else ["Число циклов = 200; Потеря массы ≤ 5%; не ссылаться на ГОСТ 10060 (для заполнителя)"]
    }
