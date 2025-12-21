import re
from dataclasses import dataclass
from typing import List, Pattern

@dataclass
class TopicRule:
    name: str
    query_patterns: List[Pattern]
    allowed_gosts: List[str]
    forbidden_gosts: List[str] | None = None

TOPIC_RULES = [
    TopicRule(
        name="dust_clay_in_coarse_agg",  # пылевидные/глинистые для щебня/гравия
        query_patterns=[re.compile(r"(пылевидн|глинист)[\w\s-]+(щебн|грави)", re.I)],
        allowed_gosts=["ГОСТ 8269.0-97", "GOST-8269.0-1997"],
        forbidden_gosts=["ГОСТ 8735-88", "GOST-8735-1988", "ГОСТ 10060"]
    ),
    TopicRule(
        name="sampling_on_conveyor_coarse",
        query_patterns=[re.compile(r"(отбор|проб)[\w\s-]+(конвейер|ленточн)", re.I)],
        allowed_gosts=["ГОСТ 8269.0-97", "GOST-8269.0-1997"],
        forbidden_gosts=["ГОСТ 8735-88", "GOST 10060"]
    ),
    TopicRule(
        name="modulus_fineness_sand",
        query_patterns=[re.compile(r"(модул[ья]?\s+крупност)", re.I)],
        allowed_gosts=["ГОСТ 8735-88", "GOST-8735-1988"],
        forbidden_gosts=["ГОСТ 8269.0-97", "ГОСТ 26633-2015"]
    ),
    TopicRule(
        name="cement_fineness_sieve_008",
        query_patterns=[re.compile(r"(тонкост|остаток).+№\s*0?08", re.I)],
        allowed_gosts=["ГОСТ 310.2-76", "GOST-310.2-1976"],
    ),
    TopicRule(
        name="frost_agg_requirements_for_concrete",
        query_patterns=[re.compile(r"(морозостойк|F\s*200).+(щебн|грави|заполнит)", re.I)],
        allowed_gosts=["ГОСТ 26633-2015", "ГОСТ 8269.0-97"],
        forbidden_gosts=["ГОСТ 10060"]
    ),
]
