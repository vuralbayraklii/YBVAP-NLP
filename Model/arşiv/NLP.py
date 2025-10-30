# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:30:28 2025

@author: vural.bayrakli
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Set
import math
import unicodedata
import itertools
from collections import defaultdict
from ZemberekClient import ZemberekClient
from BertLMScorer import BertLMScorer

# -----------------------------
# Config & weights
# -----------------------------

@dataclass
class Weights:
    alpha_lm: float = 1.00      # LM score weight
    beta_morph: float = 0.60    # morphological validity
    gamma_domain: float = 0.40  # domain lexicon boost
    delta_edit: float = 0.30    # edit distance penalty
    eps_diacrit: float = 0.20   # diacritic penalty
    zeta_keyboard: float = 0.15 # keyboard proximity penalty

@dataclass
class Thresholds:
    tau_replace: float = 0.45   # min gain over KEEP to apply REPLACE
    tau_split: float = 0.30     # min gain to allow SPLIT
    tau_merge: float = 0.30     # min gain to allow MERGE
    beam_width: int = 5         # beam size
    lm_window: int = 3          # tokens to left/right for LM

# -----------------------------
# Utilities
# -----------------------------

TR_DIACRITIC_MAP = {
    'i':'ı', 'ı':'i', 'g':'ğ', 'ğ':'g', 's':'ş', 'ş':'s',
    'c':'ç', 'ç':'c', 'o':'ö', 'ö':'o', 'u':'ü', 'ü':'u'
}

def tr_lower(s: str) -> str:
    return s.replace('I', 'ı').replace('İ', 'i').lower()

def normalize_text(s: str) -> str:
    s = unicodedata.normalize('NFC', s)
    return tr_lower(s)

def is_diacritic_pair(a: str, b: str) -> bool:
    return (a in TR_DIACRITIC_MAP and TR_DIACRITIC_MAP[a] == b) or \
           (b in TR_DIACRITIC_MAP and TR_DIACRITIC_MAP[b] == a)

def diacritic_penalty(a: str, b: str) -> float:
    # count positions where only diacritic differs
    n = min(len(a), len(b))
    count = 0
    for i in range(n):
        if a[i] != b[i] and is_diacritic_pair(a[i], b[i]):
            count += 1
    return float(count)

# Basic weighted edit distance (diacritic and keyboard-aware light version)
KEY_NEIGHBORS = {
    # Minimal TR-Q layout hint (extend as needed)
    'q':'w', 'w':'qe', 'e':'wr', 'r':'et', 't':'ry', 'y':'tu', 'u':'yi', 'i':'uo', 'o':'ip', 'p':'o',
    'a':'s', 's':'ad', 'd':'sf', 'f':'dg', 'g':'fh', 'h':'gj', 'j':'hk', 'k':'jl', 'l':'k',
    'z':'x', 'x':'zc', 'c':'xv', 'v':'cb', 'b':'vn', 'n':'bm', 'm':'n',
    'ı':'io', 'ş':'sa', 'ğ':'gh', 'ç':'cl', 'ö':'op', 'ü':'ui'
}

def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if is_diacritic_pair(a, b):
        return 0.25
    if a in KEY_NEIGHBORS and b in KEY_NEIGHBORS[a]:
        return 0.5
    return 1.0

def weighted_edit_distance(s: str, t: str) -> float:
    s = normalize_text(s); t = normalize_text(t)
    m, n = len(s), len(t)
    dp = [[0.0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1): dp[i][0] = dp[i-1][0] + 1.0
    for j in range(1, n+1): dp[0][j] = dp[0][j-1] + 1.0
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j] + 1.0,                # deletion
                dp[i][j-1] + 1.0,                # insertion
                dp[i-1][j-1] + sub_cost(s[i-1], t[j-1])  # substitution
            )
    return dp[m][n]

def window(tokens: List[str], i: int, k: int) -> Tuple[List[str], List[str]]:
    left = tokens[max(0, i-k): i]
    right = tokens[i+1: i+1+k]
    return left, right

# -----------------------------
# External interfaces (pluggable)
# -----------------------------

class ZemberekClientIface:
    """Adapt this to your zemberek-grpc client."""
    
    def __init__(self, zemberek: ZemberekClient):
        self.zemberek = zemberek
    
    def suggest(self, token: str) -> List[str]:
        return self.zemberek.spell_suggest(token)[0]
    
    
    def analyze_valid(self, token: str) -> bool:
        return self.zemberek.spell_check(token)[0]


class LMScorerIface:
    """Language model scorer: higher is better (e.g., log-prob)."""
    def score_token(self, left: List[str], cand: str, right: List[str]) -> float:
        
        return 0.0

# -----------------------------
# Candidate generation
# -----------------------------

def generate_diacritic_variants(token: str) -> Set[str]:
    # naive diacritic toggle per char (limited to one-toggle for tractability)
    variants = set()
    for i, ch in enumerate(token):
        if ch in TR_DIACRITIC_MAP:
            alt = token[:i] + TR_DIACRITIC_MAP[ch] + token[i+1:]
            variants.add(alt)
    return variants

def split_candidates(token: str, zemb: ZemberekClientIface) -> List[Tuple[str, str]]:
    cands = []
    # avoid trivial/very short splits
    for cut in range(2, len(token)-1):
        t1, t2 = token[:cut], token[cut:]
        if zemb.analyze_valid(t1) and zemb.analyze_valid(t2):
            cands.append((t1, t2))
    return cands


def merge_candidates(t1: str, t2: str, zemb: ZemberekClientIface, concept_lex: Set[str]) -> List[str]:
    merged = t1 + t2
    cands = set([merged])
    # Add diacritic variants of merged
    cands |= generate_diacritic_variants(merged)
    # Domain lexicon hits close to merged
    for w in concept_lex:
        if abs(len(w) - len(merged)) <= 2 and weighted_edit_distance(w, merged) <= 2.0:
            cands.add(w)
    # Zemberek suggestions around merged
    for s in zemb.suggest(merged)[:5]:
        cands.add(s)
    # keep only morph-valid
    return [w for w in cands if zemb.analyze_valid(w)]

def replace_candidates(token: str, zemb: ZemberekClientIface, concept_lex: Set[str]) -> List[str]:
    base = set()
    for s in zemb.suggest(token):
        base.add(s)
    base |= generate_diacritic_variants(token)
    # domain lexicon neighbors (to bias towards domain term, e.g., sigorta)
    for w in concept_lex:
        if abs(len(w) - len(token)) <= 2 and weighted_edit_distance(w, token) <= 2.0:
            base.add(w)
    # keep morph-valid
    return [w for w in base if zemb.analyze_valid(w)]

# -----------------------------
# Scoring
# -----------------------------

class TokenScorer:
    def __init__(self, zemb: ZemberekClientIface, lm: LMScorerIface,
                 concept_lex: Set[str], weights: Weights):
        self.zemb = zemb
        self.lm = lm
        self.concept_lex = concept_lex
        self.w = weights

        # Lightweight domain context hints (expand as needed)
        self.domain_left_hints = {'pano','priz','kaçak','akım','trafo','kesici','sigorta','DG','inverter','reaktif','gerilim','arıza'}
        self.nondomain_hints = {'içmek','paket','duman','tütün','sigara'}

    def domain_boost(self, cand: str, left: List[str], right: List[str]) -> float:
        boost = self.w.gamma_domain if cand in self.concept_lex else 0.0
        # tiny co-occurrence nudge
        ctx = set([normalize_text(t) for t in (left+right)])
        if cand == 'sigorta' and len(self.domain_left_hints & ctx) > 0:
            boost += 0.15
        if cand == 'sigara' and len(self.nondomain_hints & ctx) > 0:
            boost += 0.10
        return boost

    def morph_ok(self, cand: str) -> float:
        return 1.0 if self.zemb.analyze_valid(cand) else 0.0

    def lm_score(self, left: List[str], cand: str, right: List[str]) -> float:
        return self.lm.score_token(left, cand, right)

    def edit_cost(self, cand: str, original: str) -> float:
        return weighted_edit_distance(cand, original)

    def diacritic_cost(self, cand: str, original: str) -> float:
        return diacritic_penalty(cand, original)

    def keyboard_cost(self, cand: str, original: str) -> float:
        # already partly modeled in edit distance; keep small extra penalty for length-mismatches
        return 0.3 * abs(len(cand) - len(original))

    def score_candidate(self, left: List[str], cand: str, right: List[str], original: str) -> float:
        s = 0.0
        s += self.w.alpha_lm * self.lm_score(left, cand, right)
        s += self.w.beta_morph * self.morph_ok(cand)
        s -= self.w.delta_edit * self.edit_cost(cand, original)
        s -= self.w.eps_diacrit * self.diacritic_cost(cand, original)
        s -= self.w.zeta_keyboard * self.keyboard_cost(cand, original)
        return s

# -----------------------------
# Beam search corrector
# -----------------------------

@dataclass
class BeamNode:
    i: int
    out_tokens: List[str]
    score: float

class BeamSearchCorrector:
    def __init__(self,
                 zemb: ZemberekClientIface,
                 lm: LMScorerIface,
                 concept_lex: Iterable[str],
                 weights: Optional[Weights] = None,
                 thresholds: Optional[Thresholds] = None):
        self.zemb = zemb
        self.lm = lm
        self.scorer = TokenScorer(zemb, lm, set(concept_lex), weights or Weights())
        self.th = thresholds or Thresholds()

    def _keep_score(self, left: List[str], tok: str, right: List[str]) -> float:
        # how good is the original token in context
        return self.scorer.score_candidate(left, tok, right, original=tok)

    def correct(self, tokens: List[str]) -> List[str]:
        n = len(tokens)
        beam: List[BeamNode] = [BeamNode(0, [], 0.0)]
        k = self.th.lm_window

        while beam:
            # stop when all beams reached end
            if all(b.i >= n for b in beam):
                break

            new_beam: List[BeamNode] = []

            for node in beam:
                i = node.i
                if i >= n:
                    # carry finished nodes forward unchanged
                    new_beam.append(node)
                    continue

                tok = tokens[i]
                left_ctx = node.out_tokens[-k:]
                right_ctx = tokens[i+1:i+1+k]

                # KEEP
                keep_sc = self._keep_score(left_ctx, tok, right_ctx)
                new_beam.append(BeamNode(i+1, node.out_tokens + [tok], node.score + keep_sc))

                # REPLACE
                repl_cands = replace_candidates(tok, self.zemb, self.scorer.concept_lex)
                # gate by tau_replace vs KEEP
                for cand in repl_cands:
                    cand_sc = self.scorer.score_candidate(left_ctx, cand, right_ctx, original=tok)
                    if cand_sc - keep_sc >= self.th.tau_replace:
                        new_beam.append(BeamNode(i+1, node.out_tokens + [cand], node.score + cand_sc))

                # SPLIT (token -> t1 t2)
                if len(tok) >= 5:
                    for t1, t2 in split_candidates(tok, self.zemb):
                        # Score as sequential insertions: first t1 with current right context augmented by t2,
                        # then t2 with right context
                        cand1_sc = self.scorer.score_candidate(left_ctx, t1, [t2] + right_ctx, original=tok)
                        cand2_sc = self.scorer.score_candidate(left_ctx + [t1], t2, right_ctx, original=tok)
                        total_sc = cand1_sc + cand2_sc
                        if total_sc - keep_sc >= self.th.tau_split:
                            new_beam.append(BeamNode(i+1, node.out_tokens + [t1, t2], node.score + total_sc))

                # MERGE (token_i, token_{i+1} -> one token)
                if i+1 < n:
                    t_next = tokens[i+1]
                    m_cands = merge_candidates(tok, t_next, self.zemb, self.scorer.concept_lex)
                    # baseline if we KEEP tok and then KEEP next (two-step)
                    keep2_sc = keep_sc + self._keep_score(left_ctx + [tok], t_next, tokens[i+2:i+2+k])
                    for cand in m_cands:
                        # score cand using right context starting from i+2
                        cand_sc = self.scorer.score_candidate(left_ctx, cand, tokens[i+2:i+2+k], original=tok+t_next)
                        if cand_sc - keep2_sc >= self.th.tau_merge:
                            new_beam.append(BeamNode(i+2, node.out_tokens + [cand], node.score + cand_sc))

            # prune beam
            new_beam.sort(key=lambda b: b.score, reverse=True)
            beam = new_beam[:self.th.beam_width]

            # early exit if all beams finished
            if all(b.i >= n for b in beam):
                break

        # best finished node
        best = max(beam, key=lambda b: b.score)
        return best.out_tokens

# -----------------------------
# Example usage (pseudo)
# -----------------------------

class DummyZemberek(ZemberekClientIface):
    def suggest(self, token: str) -> List[str]:
        # very naive for demo
        suggestions = {
            'sigora': ['sigorta', 'sigara'],
            'ariza': ['arıza', 'ayrıca'],
            'sigortaattı': ['sigortaattı']  # will be split
        }
        return suggestions.get(token, [])
    def analyze_valid(self, token: str) -> bool:
        # accept most lowercase words in demo
        return len(token) > 0 and token.isascii() is False or token.isalpha()

class DummyLM(LMScorerIface):
    def score_token(self, left: List[str], cand: str, right: List[str]) -> float:
        # Demo: prefer 'sigorta' near electrical terms; prefer 'sigara' near 'içiyorum'
        ctx = set([normalize_text(t) for t in (left + right)])
        base = 0.0
        if cand == 'sigorta' and len(ctx & {'pano','priz','kaçak','akım','trafo','kesici'}) > 0:
            base += 2.0
        if cand == 'sigara' and 'içiyorum' in ctx:
            base += 2.0
        if cand == 'arıza' and ('nedeniyle' in ctx or 'kesinti' in ctx):
            base += 2.0
        if cand == 'ayrıca' and ('ve' in ctx or 'ama' in ctx):
            base += 1.0
        return base

if __name__ == "__main__":
    zemb = ZemberekClient(host="localhost", port=6789)
    
    zemb_cli = ZemberekClientIface(zemb)
    
    lm =  BertLMScorer()
    
    # concept_lex = {'sigorta', 'arıza', 'trafo', 'kesici', 'gerilim'}
    
    # KAVRAM SÖZLÜĞÜ (örnek): domain kelimeleriniz – mümkünse kök hâlleri
    concept_lex = {
        "sigorta", "faz", "problem", "arıza", "topraklama",
        "klemens", "kablo", "kontakt", "kaçak", "hat", "trafo", "direk", "ayırıcı"
    }

    corrector = BeamSearchCorrector(zemb_cli, lm, concept_lex,
                                    weights=Weights(),
                                    thresholds=Thresholds(beam_width=5, tau_replace=0.45, tau_split=0.30, tau_merge=0.30))

    sent = ["pano", "sigora", "attı"]
    print("IN :", " ".join(sent))
    print("OUT:", " ".join(corrector.correct(sent)))

    sent2 = ["ve", "ariza", "sayaç", "değişimi", "planlandı"]
    print("IN :", " ".join(sent2))
    print("OUT:", " ".join(corrector.correct(sent2)))

    sent3 = ["sigortaattı"]
    print("IN :", " ".join(sent3))
    print("OUT:", " ".join(corrector.correct(sent3)))
