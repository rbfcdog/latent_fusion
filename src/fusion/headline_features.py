"""Bounded, title-only facts for news tokens; cues never identify the ticker's role."""

import re

import numpy as np


FEATURE_NAMES: tuple[str, ...] = (
    "event_earnings_guidance", "event_ma", "event_legal_regulatory",
    "event_rating", "event_product", "event_other",
    "earnings_beat", "earnings_miss", "guidance_raise", "guidance_cut",
    "rating_upgrade", "rating_downgrade", "reported_pct_magnitude",
    "reported_pct_signed", "update", "rumor", "uncertainty",
)

_FINANCIAL = r"(?:earnings|eps|revenue|sales|profit|income|results)"
_METRIC = r"(?:revenue|sales|profits?|net income|earnings|eps|cash flow|margins?)"
_PCT = r"(?<![\d.])(\d+(?:\.\d+)?|\.\d+)\s*(?:%|percent\b)"
_UP = r"(?:rises?|rose|increases?|increased|gains?|gained|grew|grows?|jumps?|jumped|climbs?|climbed|up)"
_DOWN = r"(?:falls?|fell|declines?|declined|drops?|dropped|decreases?|decreased|slides?|slid|down)"

_EARNINGS = re.compile(r"\b(?:earnings|eps|quarterly results|q[1-4] results|revenue|sales|profits?|guidance|financial outlook)\b")
_PREVIEW = re.compile(r"\b(?:preview|transcript|conference call|earnings call|earnings season|to report earnings|ahead of earnings|earnings calendar)\b")
_MA = re.compile(r"\b(?:acquires?|acquired|acquisition|mergers?|takeovers?|buyouts?|to acquire|buys|bought by|being bought)\b")
_LEGAL = re.compile(r"\b(?:lawsuits?|sued|sues|litigation|settlement|antitrust|regulators?|regulatory|investigation|fda|sec|ftc)\b")
_RATING = re.compile(r"\bratings?\b")
_PRODUCT = re.compile(r"\b(?:products?|launch(?:es|ed)?|unveils?|unveiled|releases?|released|recalls?|recalled)\b")
_BEAT = re.compile(r"\b(?:beats?|beat|tops?|topped|exceeds?|exceeded)\s+(?:(?:analysts?|wall street|consensus|quarterly|q[1-4]|earnings|eps|revenue|sales|profit)\s+){0,3}(?:estimates?|expectations?|forecasts?)\b|\b(?:estimates?|expectations?)\s+(?:beaten|topped|exceeded)\b")
_MISS = re.compile(r"\b(?:miss(?:es|ed)?|below|short of)\s+(?:(?:analysts?|wall street|consensus|quarterly|q[1-4]|earnings|eps|revenue|sales|profit)\s+){0,3}(?:estimates?|expectations?|forecasts?)\b|\b(?:falls?|fell)\s+short\s+of\s+(?:estimates?|expectations?)\b")
_GUIDANCE_UP = re.compile(r"\b(?:raises?|raised|lifts?|lifted|boosts?|boosted|increases?|increased)\s+(?:(?:its|full year|annual|quarterly|fy\d+|\d{2,4}|earnings|profit|revenue|sales)\s+){0,3}(?:guidance|outlook|forecast)\b|\b(?:guidance|outlook|forecast)\s+(?:raised|lifted|boosted|increased)\b")
_GUIDANCE_DOWN = re.compile(r"\b(?:cuts?|cut|lowers?|lowered|reduces?|reduced|slashes?|slashed)\s+(?:(?:its|full year|annual|quarterly|fy\d+|\d{2,4}|earnings|profit|revenue|sales)\s+){0,3}(?:guidance|outlook|forecast)\b|\b(?:guidance|outlook|forecast)\s+(?:cut|lowered|reduced|slashed)\b")
_RATING_UP = re.compile(r"\b(?:analysts?\s+upgrades?|(?:stock|shares|rating)\s+(?:is\s+)?upgraded|upgraded?\s+to\s+(?:buy|outperform|overweight))\b")
_RATING_DOWN = re.compile(r"\b(?:analysts?\s+downgrades?|(?:stock|shares|rating)\s+(?:is\s+)?downgraded|downgraded?\s+to\s+(?:sell|underperform|underweight))\b")
_UPDATE = re.compile(r"\b(?:update|updated|updates|revised|revision|correction|corrected)\b")
_RUMOR = re.compile(r"\b(?:rumou?rs?|rumou?red|reportedly|sources say|in talks|unconfirmed)\b")
_UNCERTAIN = re.compile(r"\b(?:could|might|may(?!\s+(?:\d{4}|revenue|sales|earnings|profit|net income|eps|results|guidance|outlook)\b)|possibly|potential(?:ly)?|expected|expects|estimated|alleged|reportedly|rumou?rs?|rumou?red|unconfirmed|in talks)\b|\?")
_PERCENT = re.compile(_PCT)
_SIGNED = re.compile(rf"\b{_METRIC}\s+(?:is\s+|was\s+)?{_UP}\s+(?:by\s+)?{_PCT}|\b{_METRIC}\s+(?:is\s+|was\s+)?{_DOWN}\s+(?:by\s+)?{_PCT}|\b{_UP}\s+(?:by\s+)?{_PCT}\s+(?:in\s+)?{_METRIC}\b|\b{_DOWN}\s+(?:by\s+)?{_PCT}\s+(?:in\s+)?{_METRIC}\b")
_NEGATION = {"not", "no", "never", "without", "denies", "denied", "doesn", "didn", "isn", "wasn", "won", "cannot"}


def _affirmed(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        preceding = re.findall(r"[a-z]+", text[max(0, match.start() - 35):match.start()])[-3:]
        if not _NEGATION.intersection(preceding):
            return True
    return False


def extract_headline_features(text: str) -> np.ndarray:
    """Extract headline-level, non-issuer-attributed facts, never article-body facts.

    A percentage's sign requires a nearby explicit *financial metric* and
    change verb; an offer premium, price move or bare EPS number has no sign.
    """
    if not isinstance(text, str):
        raise TypeError("headline text must be a string")
    title = re.sub(r"[^a-z0-9.%?$]+", " ", text.lower()).strip()
    result = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    if not title:
        result[5] = 1
        return result

    preview = bool(_PREVIEW.search(title))
    earnings = bool(_EARNINGS.search(title)) and not preview
    event = next((index for index, found in enumerate((
        earnings, bool(_MA.search(title)), bool(_LEGAL.search(title)),
        bool(_RATING.search(title) or _RATING_UP.search(title) or _RATING_DOWN.search(title)),
        bool(_PRODUCT.search(title)),
    )) if found), 5)
    result[event] = 1

    # Questions and two-sided/conflicting headlines cannot give an affirmative cue.
    if not preview and not _UNCERTAIN.search(title):
        cues = [earnings and _affirmed(pattern, title) and bool(re.search(rf"\b{_FINANCIAL}\b", title))
                for pattern in (_BEAT, _MISS)]
        cues += [_affirmed(_GUIDANCE_UP, title), _affirmed(_GUIDANCE_DOWN, title),
                 _affirmed(_RATING_UP, title), _affirmed(_RATING_DOWN, title)]
        if any(cues[i] for i in (0, 2, 4)) and any(cues[i] for i in (1, 3, 5)):
            cues = [False] * 6
        result[6:12] = cues

    percentages = [min(float(m.group(1)), 100.0) / 100.0 for m in _PERCENT.finditer(title)]
    if percentages:
        result[12] = max(percentages)
    signs = set()
    if not preview and not _UNCERTAIN.search(title):
        for match in _SIGNED.finditer(title):
            if _NEGATION.intersection(re.findall(r"[a-z]+", title[max(0, match.start() - 35):match.start()])[-3:]):
                continue
            # Alternatives are metric-up, metric-down, up-percent-metric, down-percent-metric.
            group = next(i for i, value in enumerate(match.groups()) if value is not None)
            signs.add((1 if group in (0, 2) else -1) * min(float(match.group(group + 1)), 100.0) / 100.0)
        if signs and (all(x > 0 for x in signs) or all(x < 0 for x in signs)):
            result[13] = max(signs, key=abs)
    result[14:] = bool(_UPDATE.search(title)), bool(_RUMOR.search(title)), bool(_UNCERTAIN.search(title))
    return result
