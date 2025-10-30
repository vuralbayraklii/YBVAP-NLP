# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 11:03:46 2025

@author: vural.bayrakli
"""

# -*- coding: utf-8 -*-
"""
Geliştirilmiş Türkçe Spell Checker
Zemberek gRPC servisini kullanarak beam search ile düzeltme yapar
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional, Set
import math
import unicodedata
import itertools
from collections import defaultdict
from ZemberekClient import ZemberekClient
from Fonksiyonlar import *
import pandas as pd
from idBasedCategoryMatcher import IDBasedCategoryMatcher
import json
from config import NLPConfig
from Levenshtein import distance
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from arıza_işleme import *


# -----------------------------
# Değişkenler
# -----------------------------
config_path = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\config\config.json"

# Kullanım
config = NLPConfig.from_json(config_path, "vural.bayrakli")

print(config.host)
print(config.port)
print(config.dokunulmayacak_kelimeler_path)
print(config.arızalar_path)
print(config.input_concatted_path)

# -----------------------------
# Stop Words
# -----------------------------

STOP_WORDS = {
    # Çok genel terimler
    'arıza',        # 24 - Her kategoride var       
    'etmek',        # 99 - Yardımcı fiil
    'etki',         # 29 - Çok genel
    'vermek',       # 35 - Yardımcı fiil
    'olmak',        # Listede yok ama eklenebilir
    'yapmak',       # Listede yok ama eklenebilir
    
    # Bağlaçlar ve edatlar
    've',           # 303
    'ile',          # Listede yok
    'veya',         # Listede yok
    'için',         # Listede yok
    
    # Varlık/Yokluk belirteçleri
    'var',          # 226
    'yok',          # 212
    
    # Çok genel fiiller
    'almak',        # 181 - Çok genel
    'çıkmak',       # 126 - Çok genel
    'gelmek',       # 293 - Çok genel
    'girmek',       # 276 - Çok genel
    'vermek',       # 35 - Çok genel
    'görmek',       # 304 - Çok genel
    'tutmak',       # 294 - Çok genel
    'duymak',       # 203 - Çok genel
    
    # Genel sıfatlar
    'yüksek',       # 115 - Çok genel
    'düşük',        # 9 - Çok genel
    'aşırı',        # 40 - Çok genel
    'yanlış',       # 73 - Çok genel
    'bozuk',        # 154 - Çok genel
    'zayıf',        # 69 - Çok genel
    'yetersiz',     # 103 - Çok genel
    'tam',          # 260 - Çok genel
    
    # Yön/Yer belirteçleri (çok genel)
    'iç',           # 152 - Çok genel
    'dış',          # 261 - Çok genel
    'üst',          # 300 - Çok genel
    'üzeri',        # 264 - Çok genel
    'yer',          # 125 - Çok genel
    'yerinde',      # 184 - Çok genel
    
    # Zaman/Durum belirteçleri
    'sonra',        # 82 - Çok genel
    'zaman',        # 295 - Çok genel
    'tekrar',       # 207 - Çok genel
    
    # Genel kavramlar
    'sorun',        # 89 - Çok genel (arıza ile eş anlamlı)
    'hata',         # 144 - Çok genel (arıza ile eş anlamlı)
    'sebep',        # 107 - Çok genel
    'neden',        # 237 - Çok genel
    'kalite',       # 67 - Çok genel
    'fark',         # 91 - Çok genel
    'değer',        # 285 - Çok genel
    'test',         # 240 - Çok genel
    
    # Belirsiz terimler
    'artmak',       # 231 - Çok genel
    'düşmek',       # 266 - Çok genel
    'yükselmek',    # 219 - Çok genel
    'oturmak',      # 245 - Belirsiz
    'çalışmak',     # 232 - Çok genel
    'atlamak',      # 217 - Belirsiz
    
    # Ölçüm/İzleme terimleri (genel)
    'ölçüm',        # 224 - Çok genel
    'izlemek',      # 284 - Çok genel
    'tespit',       # 56 - Çok genel
    'kontrol',      # 39 - Çok genel
}   
    
# -----------------------------
# Config & weights
# -----------------------------

@dataclass
class Weights:
    alpha_lm: float = 0.50      # 1.00'dan düşürüldü (artık skorlar büyük)
    beta_morph: float = 0.80    
    gamma_domain: float = 0.50  
    delta_edit: float = 0.25    
    eps_diacrit: float = 0.15   
    zeta_keyboard: float = 0.10 
    eta_split: float = 0.5

@dataclass
class Thresholds:
    tau_replace: float = 0.30   # min gain over KEEP to apply REPLACE (düşürüldü)
    tau_split: float = 0.20     # min gain to allow SPLIT (düşürüldü)
    tau_merge: float = 0.25     # min gain to allow MERGE (düşürüldü)
    beam_width: int = 10        # beam size (artırıldı)
    lm_window: int = 2         # tokens to left/right for LM
    min_token_len_for_split: int = 4  # minimum token length to consider splitting


# -----------------------------
# External interfaces
# -----------------------------

class ZemberekClientIface:
    """Zemberek gRPC client adaptörü"""
    
    def __init__(self, zemberek: ZemberekClient):
        self.zemberek = zemberek
        self._cache_valid = {}
        self._cache_suggest = {}
    
    
    def suggest(self, token: str) -> List[str]:
        """Kelime önerileri al (cache'li)"""
        if token not in self._cache_suggest:
            try:
                suggestions, _ = self.zemberek.spell_suggest(token)
                self._cache_suggest[token] = suggestions if suggestions else []
            except:
                self._cache_suggest[token] = []
        return self._cache_suggest[token]
    
    def analyze_valid(self, word: str) -> bool:
        """
        Kelimenin geçerli Türkçe olup olmadığını kontrol et
        - Hem küçük hem büyük harfle kontrol eder
        
        Args:
            word: Kontrol edilecek kelime
        
        Returns:
            bool: Geçerli ise True
        """
        if not word:
            return False
        
        # 1. Küçük harfle kontrol et (normal kelimeler için)
        is_valid_lower, _ = self.zemberek.spell_check(word.lower())
        if is_valid_lower:
            return True
        
        # 2. İlk harf büyük (özel isimler için)
        is_valid_capitalized, _ = self.zemberek.spell_check(word.capitalize())
        if is_valid_capitalized:
            return True
        
        # 3. Tamamı büyük (kısaltmalar için)
        is_valid_upper, _ = self.zemberek.spell_check(word.upper())
        if is_valid_upper:
            return True
        
        # 4. Türkçe karakterlerle capitalize (İ/i problemi için)
        capitalized_tr = self._turkish_capitalize(word)
        is_valid_tr_cap, _ = self.zemberek.spell_check(capitalized_tr)
        if is_valid_tr_cap:
            return True
        
        return False
    
    
    def _turkish_capitalize(self, word: str) -> str:
        """
        Türkçe karakterlere duyarlı capitalize
        
        Args:
            word: Capitalize edilecek kelime
        
        Returns:
            Capitalize edilmiş kelime
        """
        if not word:
            return word
        
        # İlk karakteri büyüt (Türkçe kurallarına göre)
        first_char = word[0]
        
        # Türkçe özel karakterler
        turkish_lower_to_upper = {
            'i': 'İ',
            'ı': 'I',
            'ğ': 'Ğ',
            'ü': 'Ü',
            'ş': 'Ş',
            'ö': 'Ö',
            'ç': 'Ç'
        }
        
        if first_char in turkish_lower_to_upper:
            first_upper = turkish_lower_to_upper[first_char]
        else:
            first_upper = first_char.upper()
        
        return first_upper + word[1:].lower()
        
    def normalize(self, token: str) -> str:
        """Kelimeyi normalize et"""
        try:
            normalized, _ = self.zemberek.normalize_text(token)
            return normalized if normalized else token
        except:
            return token
        
    def tokenize(self, text: str) -> List[str]:
        """Metni tokenlara ayır"""
        try:
            tokens, _ = self.zemberek.tokenize(text)
            return tokens if tokens else text.split()
        except:
            return text.split()
        
    def get_lemma(self, text):
        """ Kelimenin kökünü dönder """
        sonuclar = self.zemberek.analyze_sentence(text)
        
        return tr_lower(sonuclar[0]["lemmas"][0])

class LMScorerIface:
    """Language model scorer interface"""
    def score_token(self, left: List[str], cand: str, right: List[str]) -> float:
        # Bu BertLMScorer'a bağlanacak
        return 0.0

# -----------------------------
# Scoring
# -----------------------------
class TokenScorer:
    """Token skorlama sınıfı"""
    
    def __init__(self, zemb: ZemberekClientIface, lm: LMScorerIface,
                 concept_lex: Set[str], weights: Weights):
        self.zemb = zemb
        self.lm = lm
        self.concept_lex = set(tr_lower(w) for w in concept_lex)
        self.w = weights

        # Domain context hints (elektrik terimleri)
        self.domain_hints = {
            'pano', 'priz', 'kaçak', 'akım', 'trafo', 'kesici', 'sigorta', 
            'dg', 'inverter', 'reaktif', 'gerilim', 'arıza', 'faz', 'nötr',
            'toprak', 'kablo', 'hat', 'direk', 'sayaç', 'kontaktör', 'şalter',
            'elektrik', 'voltaj', 'amper', 'watt', 'güç', 'enerji'
        }
        
        # Non-domain hints (sigara ile ilgili)
        self.nondomain_hints = {
            'içmek', 'içiyorum', 'paket', 'duman', 'tütün', 'sigara',
            'yakıyorum', 'bıraktım', 'zararlı', 'sağlık'
        }

    def domain_boost(self, cand: str, left: List[str], right: List[str]) -> float:
        """Domain sözlüğü ve context'e göre bonus"""
        cand_lower = tr_lower(cand)
        boost = 0.0
        
        # Domain sözlüğünde varsa bonus
        if cand_lower in self.concept_lex:
            boost += self.w.gamma_domain
        
        # # Context'e göre ek bonus
        # ctx = set(tr_lower(t) for t in (left + right))
        
        # # Elektrik domain'i için
        # if cand_lower == 'sigorta':
        #     domain_overlap = len(self.domain_hints & ctx)
        #     if domain_overlap > 0:
        #         boost += 0.3 * domain_overlap
        
        # # Sigara domain'i için
        # elif cand_lower == 'sigara':
        #     nondomain_overlap = len(self.nondomain_hints & ctx)
        #     if nondomain_overlap > 0:
        #         boost += 0.2 * nondomain_overlap
        
        # # Arıza kelimesi için
        # elif cand_lower == 'arıza':
        #     if any(w in ctx for w in ['nedeniyle', 'kesinti', 'elektrik', 'sayaç']):
        #         boost += 0.3
        
        return boost

    def morph_ok(self, cand: str) -> float:
        """Morfolojik geçerlilik skoru"""
        return self.w.beta_morph if self.zemb.analyze_valid(cand) else self.w.beta_morph * (-1)

    def lm_score(self, left: List[str], cand: str, right: List[str]) -> float:
        """Language model skoru"""
        return self.lm.score_token(left, cand, right)

    def edit_cost(self, cand: str, original: str) -> float:
        """Edit distance maliyeti"""
        dist = weighted_edit_distance(cand, original)
        # Normalize et (0-1 arası)
        max_len = max(len(cand), len(original))
        if max_len > 0:
            return dist / max_len
        return 0.0

    def diacritic_cost(self, cand: str, original: str) -> float:
        """Diacritic farklılık maliyeti"""
        return diacritic_penalty(cand, original)

    def keyboard_cost(self, cand: str, original: str) -> float:
        """Klavye uzaklık maliyeti"""
        # Edit distance'da zaten hesaplanıyor, ekstra küçük ceza
        return 0.1 * abs(len(cand) - len(original))

    def score_candidate(self, left: List[str], cand: str, right: List[str], 
                        original: str, operation: str = 'REPLACE') -> float:
        """Aday kelimeyi skorla"""
        s = 0.0
        
        # Temel skorlar
        # s += self.w.alpha_lm * self.lm_score(left, cand, right)
        s += self.morph_ok(cand)
        # s += self.domain_boost(cand, left, right)
        
        # Maliyetler
        s -= self.w.delta_edit * self.edit_cost(cand, original)
        s -= self.w.eps_diacrit * self.diacritic_cost(cand, original)
        s -= self.w.zeta_keyboard * self.keyboard_cost(cand, original)
        
        # # İşlem bonusu/cezası
        # if operation == 'SPLIT':
        #     s += self.w.eta_split  # Split işlemine bonus
        # elif operation == 'MERGE':
        #     s += 0.2  # Merge işlemine küçük bonus
        
        return s


# -----------------------------
# Beam search corrector
# -----------------------------

from dataclasses import dataclass
from typing import List, Optional, Iterable, Set, Tuple


@dataclass
class BeamNode:
    """Beam search node'u - Quality score ile"""
    i: int  # Mevcut pozisyon
    out_tokens: List[str] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    quality_score: float = 0.0  # ✨ YENİ: Node kalite skoru
    
    def __repr__(self):
        return (f"BeamNode(i={self.i}, "
                f"tokens={len(self.out_tokens)}, "
                f"quality={self.quality_score:.2f})")

class BeamSearchCorrector:
    """Beam search ile düzeltme yapan ana sınıf"""
    
    def __init__(self,
                 zemb: ZemberekClientIface,
                 do_not_touch: Iterable[str],
                 lemma2id,
                 weights: Optional[Weights] = None,
                 thresholds: Optional[Thresholds] = None):
        self.zemb = zemb
        self.do_not_touch = set(do_not_touch) if do_not_touch is not None else set()
        self.th = thresholds or Thresholds()
        self.lemma2id = lemma2id
    
    def arıza_içinde_v2(self, tok):
        return tok in self.lemma2id

    def _node_process(
        self, 
        tok: str, 
        node: BeamNode, 
        i: int, 
        tokens: List[str], 
        n: int
    ) -> List[BeamNode]:
        """
        Bir token için tüm olası işlemleri uygula
        - ✨ YENİ: Quality score hesaplama
        - ✨ YENİ: Early return stratejisi
        """
        candidates: List[BeamNode] = []
        
        # Temel bilgiler
        tok_lemma = self.zemb.get_lemma(tok) if self.zemb.get_lemma(tok) != "unk" else tok
        is_tok_concept = self.arıza_içinde_v2(tok) or self.arıza_içinde_v2(tok_lemma)
        is_tok_protected = tok in self.do_not_touch
        is_tok_valid = self.zemb.analyze_valid(tok)
        
        # ============================================================
        # 1. CONCEPT / PROTECTED KELİME - En yüksek öncelik
        # ============================================================
        if is_tok_concept:
            candidates.append(BeamNode(
                i=i + 1,
                out_tokens=node.out_tokens + [tok_lemma if tok_lemma != "unk" else tok],
                operations=node.operations + ['KEEP(CONCEPT)'],
                quality_score=node.quality_score + 2.0  # ✨ +2.0 bonus
            ))
            return candidates  # ✨ EARLY RETURN
        
        if is_tok_protected:
            candidates.append(BeamNode(
                i=i + 1,
                out_tokens=node.out_tokens + [tok],
                operations=node.operations + ['KEEP(PROTECTED)'],
                quality_score=node.quality_score + 1.5  # ✨ +1.5 bonus
            ))
            return candidates  # ✨ EARLY RETURN
        
        # ============================================================
        # 2. LEVENSHTEIN - Yakın concept kelimeleri
        # ============================================================
        if len(tok) > 3:
            lev_candidates = []
            
            for w in self.lemma2id:
                if abs(len(w) - len(tok)) <= 3:
                    dist = distance(w, tok)
                    
                    if dist <= 2.5:
                        w_lemma = self.zemb.get_lemma(w)
                        
                        if self.arıza_içinde_v2(w_lemma):
                            # ✨ Quality: Mesafe càng küçük, quality càng yüksek
                            quality_bonus = 2.0 - (dist * 0.3)
                            lev_candidates.append((dist, w, w_lemma, quality_bonus))
            
            if lev_candidates:
                # En iyi 3 adayı al
                lev_candidates.sort(key=lambda x: x[0])
                
                for dist, w, w_lemma, quality_bonus in lev_candidates[:3]:
                    candidates.append(BeamNode(
                        i=i + 1,
                        out_tokens=node.out_tokens + [w_lemma],
                        operations=node.operations + [f'LEVENSHTEIN(CONCEPT:{tok}->{w},d={dist:.1f})'],
                        quality_score=node.quality_score + quality_bonus  # ✨ Quality
                    ))
                
                return candidates  # ✨ EARLY RETURN
        
        # ============================================================
        # 3. SPLIT - Kelimeyi böl
        # ============================================================
        if len(tok) >= self.th.min_token_len_for_split and not is_tok_valid:
            split_cands = split_candidates(tok, self.zemb, self.lemma2id)
            
            found_split = False
            for t1, t2 in split_cands[:3]:
                t1_lemma = self.zemb.get_lemma(t1)
                t2_lemma = self.zemb.get_lemma(t2)
                
                t1_is_concept = self.arıza_içinde_v2(t1) or self.arıza_içinde_v2(t1_lemma)
                t2_is_concept = self.arıza_içinde_v2(t2) or self.arıza_içinde_v2(t2_lemma)
                
                if not (t1_is_concept or t2_is_concept):
                    continue
                
                valid_tokens = []
                concept_count = 0
                
                if t1_is_concept:
                    valid_tokens.append(t1_lemma)
                    concept_count += 1
                if t2_is_concept:
                    valid_tokens.append(t2_lemma)
                    concept_count += 1
                
                if valid_tokens:
                    # ✨ Quality: İki parça da concept ise yüksek bonus
                    quality_bonus = 1.5 * concept_count
                    
                    candidates.append(BeamNode(
                        i=i + 1,
                        out_tokens=node.out_tokens + valid_tokens,
                        operations=node.operations + [f'SPLIT(CONCEPT:{tok}->{t1}+{t2})'],
                        quality_score=node.quality_score + quality_bonus  # ✨ Quality
                    ))
                    found_split = True
            
            if found_split:
                return candidates  # ✨ EARLY RETURN
        
        # ============================================================
        # 4. MERGE - Sonraki token ile birleştir
        # ============================================================
        if i + 1 < n:
            t_next = tokens[i + 1]
            m_cands = merge_candidates(tok, t_next, self.zemb, self.lemma2id.keys())
            
            found_merge = False
            for cand in m_cands[:3]:
                cand_lemma = self.zemb.get_lemma(cand)
                
                if self.arıza_içinde_v2(cand) or self.arıza_içinde_v2(cand_lemma):
                    candidates.append(BeamNode(
                        i=i + 2,  # İki token birleşti
                        out_tokens=node.out_tokens + [cand_lemma],
                        operations=node.operations + [f'MERGE(CONCEPT:{tok}+{t_next}->{cand})'],
                        quality_score=node.quality_score + 1.8  # ✨ +1.8 bonus
                    ))
                    found_merge = True
            
            if found_merge:
                return candidates  # ✨ EARLY RETURN
        
        # ============================================================
        # 5. REPLACE - Benzer kelime ile değiştir
        # ============================================================
        repl_cands = replace_candidates(tok, self.zemb, self.lemma2id)
        
        found_replace = False
        for cand in repl_cands[:5]:
            cand_lemma = self.zemb.get_lemma(cand)
            
            if self.arıza_içinde_v2(cand) or self.arıza_içinde_v2(cand_lemma):
                candidates.append(BeamNode(
                    i=i + 1,
                    out_tokens=node.out_tokens + [cand_lemma],
                    operations=node.operations + [f'REPLACE(CONCEPT:{tok}->{cand})'],
                    quality_score=node.quality_score + 1.5  # ✨ +1.5 bonus
                ))
                found_replace = True
        
        if found_replace:
            return candidates  # ✨ EARLY RETURN
        
        # ============================================================
        # 6. KEEP - Hiçbir işlem yapma (son çare)
        # ============================================================
        # ✨ Quality: Geçerli kelime ise +0.5, değilse -0.5
        quality_change = 0.5 if is_tok_valid else -0.5
        
        candidates.append(BeamNode(
            i=i + 1,
            out_tokens=node.out_tokens + [tok],
            operations=node.operations + ['KEEP(VALID)' if is_tok_valid else 'KEEP(INVALID)'],
            quality_score=node.quality_score + quality_change  # ✨ Quality
        ))
        
        return candidates    

    def correct(self, tokens: List[str], verbose: bool = False) -> List[BeamNode]:
        """
        Token listesini düzelt - 3 paralel strateji ile
        
        NODE 1: NORMALIZE PATH
        - Her tokeni normalize et
        - Normalize edilmiş tokenin KÖKÜNÜ al
        
        NODE 2: DICTIONARY PATH
        - Sözlükte varsa → KÖKÜNÜ al
        - Sözlükte yoksa ama geçerliyse → TOKENIN KENDİSİNİ al
        - Sözlükte yok + geçersizse → Düzelt (normalize/split/merge/suggest) ve ilk sonucun KÖKÜNÜ al
        
        NODE 3: NON-DICTIONARY PATH
        - Geçerliyse → TOKENIN KENDİSİNİ al
        - Geçersizse → Düzelt ve ilk sonucun KÖKÜNÜ al
        
        Returns:
            3 BeamNode listesi
        """
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📝 Original tokens: {tokens}")
            print(f"{'='*80}")
        
        # ============================================================
        # NODE 1: NORMALIZE PATH
        # ============================================================
        node1 = self._create_node1_normalize(tokens, verbose)
        
        # ============================================================
        # NODE 2: DICTIONARY PATH
        # ============================================================
        node2 = self._create_node2_dictionary(tokens, verbose)
        
        # ============================================================
        # NODE 3: NON-DICTIONARY PATH
        # ============================================================
        # node3 = self._create_node3_non_dictionary(tokens, verbose)
        
        # ============================================================
        # Sonuçları döndür
        # ============================================================
        all_nodes = [node1, node2]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🏆 FINAL RESULTS - 3 Strategies")
            print(f"{'='*80}")
            
            for i, node in enumerate(all_nodes, 1):
                print(f"\nNODE {i}: {node.operations[0]}")
                print(f"   Quality: {node.quality_score:.2f}")
                print(f"   Tokens: {' '.join(node.out_tokens)}")
                print(f"   Token Count: {len(node.out_tokens)}")
        
        return all_nodes
    
    def _create_node1_normalize(self, tokens: List[str], verbose: bool = False) -> BeamNode:
        """
        ✨ NODE 1: NORMALIZE PATH
        
        Strateji:
        - Her tokeni normalize et
        - Normalize edilmiş tokenin KÖKÜNÜ al
        
        Returns:
            BeamNode
        """
        if verbose:
            print(f"\n🔵 NODE 1: NORMALIZE PATH")
        
        out_tokens = []
        operations = ['STRATEGY:NODE1(NORMALIZE_ALL)']
        quality_score = 0.0
        
        for tok in tokens:
            tok_lower = tok.lower()
            
            # Normalize et
            normalized = self.zemb.normalize(tok_lower)
            
            # Normalize edilmiş tokenin KÖKÜNÜ al
            norm_lemma = self.zemb.get_lemma(normalized)
            
            # Kök UNK değilse kökü al, yoksa normalize edilmiş hali
            final_token = norm_lemma if norm_lemma != "UNK" else normalized
            
            out_tokens.append(final_token)
            
            # Quality hesapla
            if normalized != tok_lower:
                operations.append(f'NORMALIZE({tok_lower}→{normalized}→{final_token})')
                
                # Sözlükte mi?
                if self.arıza_içinde_v2(final_token):
                    quality_score += 2.0  # Normalize + concept
                elif self.zemb.analyze_valid(final_token):
                    quality_score += 1.0  # Normalize + valid
                else:
                    quality_score += 0.5  # Normalize edildi
            else:
                operations.append(f'KEEP({tok_lower}→{final_token})')
                
                if self.arıza_içinde_v2(final_token):
                    quality_score += 1.5  # Zaten concept
                elif self.zemb.analyze_valid(tok_lower):
                    quality_score += 1  # Zaten valid
                else:
                    quality_score += 0.0  # Değişmedi + invalid
        
        if verbose:
            print(f"   Quality: {quality_score:.2f}")
            print(f"   Sample: {' '.join(out_tokens[:5])}...")
        
        return BeamNode(
            i=len(tokens),
            out_tokens=out_tokens,
            operations=operations,
            quality_score=quality_score
        )
    
    def _create_node2_dictionary(self, tokens: List[str], verbose: bool = False) -> BeamNode:
        """
        ✨ NODE 2: DICTIONARY PATH (Sıralı İşlem + Skorlama)
        
        Strateji:
        1. Her token için sırayla:
           - Sözlükte varsa → KÖKÜNÜ al (yüksek skor)
           - Sözlükte yoksa → Düzeltme işlemleri yap
             - Sözlükte bulundu mu? → KÖKÜNÜ al (orta skor)
             - Bulunamadı mı? → Orijinali al (geçerlilik durumuna göre skorla)
        
        Returns:
            BeamNode
        """
        if verbose:
            print(f"\n🟢 NODE 2: DICTIONARY PATH")
        
        out_tokens = []
        operations = ['STRATEGY:NODE2(DICTIONARY_PRIORITY)']
        quality_score = 0.0
        
        i = 0
        while i < len(tokens):
            tok = tokens[i].lower()
            
            if verbose:
                print(f"   Processing token[{i}]: '{tok}'")
            
            # Token işleme
            candidates = self._node_process_node2(tok, i, tokens, verbose)
            
            if candidates:
                # En iyi adayı seç (en yüksek quality_score)
                best = max(candidates, key=lambda x: x.get('quality_score', 0))
                
                out_tokens.append(best['token'])
                operations.append(best['operation'])
                quality_score += best['quality_score']
                i += best['consumed']
                
                if verbose:
                    print(f"      → {best['token']} (score: +{best['quality_score']:.1f})")
            else:
                # Fallback (olmamalı ama güvenlik için)
                out_tokens.append(tok)
                operations.append(f'FALLBACK({tok})')
                quality_score -= 1.0
                i += 1
        
        if verbose:
            print(f"   Quality: {quality_score:.2f}")
            print(f"   Tokens: {out_tokens}")
        
        return BeamNode(
            i=len(tokens),
            out_tokens=out_tokens,
            operations=operations,
            quality_score=quality_score
        )
    
    def _node_process_node2(
        self,
        tok: str,
        i: int,
        tokens: List[str],
        verbose: bool = False
    ) -> List[Dict]:
        """NODE 2 için token işleme"""
        
        n = len(tokens)
        candidates = []
        
        # ============================================================
        # Helper: Güvenli lemma alma
        # ============================================================
        def safe_get_lemma(word):
            """Lemma al, UNK ise orijinali döndür"""
            lemma = self.zemb.get_lemma(word)
            if not lemma or lemma.lower() == "unk":
                return word
            return lemma
        
        # ============================================================
        # 0. TOKEN ZATEN SÖZLÜKTE Mİ?
        # ============================================================
        tok_lemma = safe_get_lemma(tok)  # ✅ Güvenli
        is_tok_concept = self.arıza_içinde_v2(tok) or self.arıza_içinde_v2(tok_lemma)
        is_tok_protected = tok in self.do_not_touch
        
        if is_tok_concept:
            # ✅ Sözlükte (concept), KÖKÜNÜ al
            candidates.append({
                'token': tok_lemma,  # ✅ Artık UNK olmayacak
                'consumed': 1,
                'operation': f'DICT→LEMMA({tok}→{tok_lemma})',
                'quality_score': 5.0
            })
            return candidates
        
        if is_tok_protected:
            candidates.append({
                'token': tok,
                'consumed': 1,
                'operation': f'PROTECTED({tok})',
                'quality_score': 4.5
            })
            return candidates
        
        # ============================================================
        # 1. LEVENSHTEIN
        # ============================================================
        if len(tok) > 3:
            best_lev_candidates = []
            
            for w in self.lemma2id:
                if abs(len(w) - len(tok)) <= 3:
                    dist = distance(w, tok)
                    
                    if dist <= 2.5:
                        w_lemma = safe_get_lemma(w)  # ✅ Güvenli
                        
                        if self.arıza_içinde_v2(w_lemma):
                            quality_bonus = 4.0 - (dist * 0.5)
                            best_lev_candidates.append({
                                'token': w_lemma,  # ✅ UNK olmayacak
                                'consumed': 1,
                                'operation': f'LEV→DICT({tok}→{w},d={dist:.1f})',
                                'quality_score': quality_bonus,
                                'dist': dist
                            })
            
            if best_lev_candidates:
                best_lev_candidates.sort(key=lambda x: x['dist'])
                return best_lev_candidates[:3]
        
        # ============================================================
        # 2. NORMALIZE
        # ============================================================
        normalized = self.zemb.normalize(tok)
        if normalized != tok:
            norm_lemma = safe_get_lemma(normalized)  # ✅ Güvenli
            
            if self.arıza_içinde_v2(normalized) or self.arıza_içinde_v2(norm_lemma):
                candidates.append({
                    'token': norm_lemma,  # ✅ UNK olmayacak
                    'consumed': 1,
                    'operation': f'NORMALIZE→DICT({tok}→{norm_lemma})',
                    'quality_score': 3.5
                })
                return candidates
            
            # Normalize + Levenshtein
            if len(normalized) > 3:
                for w in self.lemma2id:
                    if abs(len(w) - len(normalized)) <= 2:
                        dist = distance(w, normalized)
                        
                        if dist <= 2:
                            w_lemma = safe_get_lemma(w)  # ✅ Güvenli
                            if self.arıza_içinde_v2(w_lemma):
                                candidates.append({
                                    'token': w_lemma,  # ✅ UNK olmayacak
                                    'consumed': 1,
                                    'operation': f'NORMALIZE+LEV→DICT({tok}→{normalized}→{w})',
                                    'quality_score': 3.0
                                })
                                return candidates
        
        # ============================================================
        # 3. SPLIT
        # ============================================================
        if len(tok) >= self.th.min_token_len_for_split:
            split_cands = split_candidates(tok, self.zemb, self.lemma2id)
            
            for t1, t2 in split_cands[:3]:
                # t1 kontrolü
                t1_lemma = safe_get_lemma(t1)  # ✅ Güvenli
                t1_is_concept = self.arıza_içinde_v2(t1) or self.arıza_içinde_v2(t1_lemma)
                
                if t1_is_concept:
                    candidates.append({
                        'token': t1_lemma,  # ✅ UNK olmayacak
                        'consumed': 1,
                        'operation': f'SPLIT→DICT({tok}→{t1}+{t2}→{t1_lemma})',
                        'quality_score': 2.5
                    })
                
                # t2 kontrolü
                t2_lemma = safe_get_lemma(t2)  # ✅ Güvenli
                t2_is_concept = self.arıza_içinde_v2(t2) or self.arıza_içinde_v2(t2_lemma)
                
                if t2_is_concept:
                    candidates.append({
                        'token': t2_lemma,  # ✅ UNK olmayacak
                        'consumed': 1,
                        'operation': f'SPLIT→DICT({tok}→{t1}+{t2}→{t2_lemma})',
                        'quality_score': 2.5
                    })
            
            if candidates:
                return candidates
        
        # ============================================================
        # 4. MERGE
        # ============================================================
        if i + 1 < n:
            t_next = tokens[i+1].lower()
            m_cands = merge_candidates(tok, t_next, self.zemb, self.lemma2id.keys())
            
            for cand in m_cands[:3]:
                cand_lemma = safe_get_lemma(cand)  # ✅ Güvenli
                
                if self.arıza_içinde_v2(cand) or self.arıza_içinde_v2(cand_lemma):
                    candidates.append({
                        'token': cand_lemma,  # ✅ UNK olmayacak
                        'consumed': 2,
                        'operation': f'MERGE→DICT({tok}+{t_next}→{cand_lemma})',
                        'quality_score': 3.0
                    })
            
            if candidates:
                return candidates
        
        # ============================================================
        # 5. REPLACE
        # ============================================================
        repl_cands = replace_candidates(tok, self.zemb, self.lemma2id)
        
        for cand in repl_cands[:5]:
            cand_lemma = safe_get_lemma(cand)  # ✅ Güvenli
            
            if self.arıza_içinde_v2(cand) or self.arıza_içinde_v2(cand_lemma):
                candidates.append({
                    'token': cand_lemma,  # ✅ UNK olmayacak
                    'consumed': 1,
                    'operation': f'REPLACE→DICT({tok}→{cand_lemma})',
                    'quality_score': 2.8
                })
        
        if candidates:
            return candidates
        
        # ============================================================
        # 6. HİÇBİRİ BAŞARISIZ → Orijinali al
        # ============================================================
        is_tok_valid = self.zemb.analyze_valid(tok)
        
        if is_tok_valid:
            candidates.append({
                'token': tok,  # ✅ Orijinal tokeni al (UNK değil!)
                'consumed': 1,
                'operation': f'NOT_IN_DICT→KEEP_VALID({tok})',
                'quality_score': 1.0
            })
        else:
            candidates.append({
                'token': tok,  # ✅ Orijinal tokeni al (UNK değil!)
                'consumed': 1,
                'operation': f'NOT_IN_DICT→KEEP_INVALID({tok})',
                'quality_score': -1.0
            })
        
        return candidates
    
    def safe_get_lemma(self, word):
        """Güvenli lemma alma - class metodu olarak"""
        lemma = self.zemb.get_lemma(word)
        if not lemma or lemma.lower() == "unk":
            return word
        return lemma
    
    def _global_normalize(self, tokens: List[str], verbose: bool = False) -> Tuple[List[str], Dict[str, str]]:
        """
        ✨ YENİ: Tüm input'u bağlam bazlı normalize et
        
        Args:
            tokens: Original token listesi
            verbose: Debug output
        
        Returns:
            (normalized_tokens, normalization_map)
        """
        normalized = []
        normalization_map = {}
        
        for i, tok in enumerate(tokens):
            tok_lower = tok.lower()
            
            # Bağlam al (önceki ve sonraki 2 kelime)
            context_before = tokens[max(0, i-2):i]
            context_after = tokens[i+1:min(len(tokens), i+3)]
            
            # Normalize et
            norm_tok = self.zemb.normalize(tok_lower)
            
            # Bağlam bazlı düzeltmeler
            if norm_tok != tok_lower:
                # Önceki/sonraki kelimelerle uyumlu mu kontrol et
                if self._is_contextually_valid(norm_tok, context_before, context_after):
                    normalized.append(norm_tok)
                    normalization_map[tok_lower] = norm_tok
                else:
                    normalized.append(tok_lower)
            else:
                normalized.append(tok_lower)
        
        return normalized, normalization_map
    
    
    def _is_contextually_valid(self, word: str, context_before: List[str], context_after: List[str]) -> bool:
        """
        Normalize edilmiş kelimenin bağlamda geçerli olup olmadığını kontrol et
        
        Args:
            word: Kontrol edilecek kelime
            context_before: Önceki kelimeler
            context_after: Sonraki kelimeler
        
        Returns:
            bool: Geçerli ise True
        """
        # Basit kontrol: normalize edilmiş kelime geçerli Türkçe kelime mi?
        if not self.zemb.analyze_valid(word):
            return False
        
        # Daha gelişmiş: Bağlamda concept var mı kontrol et
        # Örnek: "trafo" + "yaglı" → "yağlı" normalize edilebilir
        for ctx_word in context_before + context_after:
            ctx_lemma = self.zemb.get_lemma(ctx_word.lower())
            if self.arıza_içinde_v2(ctx_lemma):
                # Bağlamda concept var, normalize güvenli
                return True
        
        return True  # Varsayılan: geçerli
    
    
    def _create_fallback_node(self, original_node: BeamNode, tok: str) -> BeamNode:
        """
        ✨ YENİ: Fallback node oluştur (düşük quality)
        """
        return BeamNode(
            i=original_node.i + 1,
            out_tokens=original_node.out_tokens + [tok],
            operations=original_node.operations + ['KEEP(FALLBACK)'],
            quality_score=original_node.quality_score - 0.5  # ✨ Ceza
        )
    

# -----------------------------
# Main test
# -----------------------------

if __name__ == "__main__":
    
    # Zemberek client'ı başlat
    print("Zemberek client başlatılıyor...")
    
    zemb = ZemberekClient(host=config.host, port=config.port)
    zemb_cli = ZemberekClientIface(zemb)
    
    dokunma = pd.read_excel(config.dokunulmayacak_kelimeler_path)["kelimeler"].values

    df = pd.read_excel(config.arızalar_path)
    
    # structures = arızaları_işle_v2(df, zemb, dokunma)
    # structures = arızaları_işle_v3(df, 
    #                                config.cause_code_şebeke_unsuru_path, 
    #                                zemb, 
    #                                dokunma)
    
    structures = arızaları_işle_v4_morphosemantic(
        df,
        config.cause_code_şebeke_unsuru_path,
        zemb,
        dokunma
    )

    _input = pd.read_excel(config.input_concatted_path)
    import random
    input_sample = _input.iloc[random.randint(1,len(_input))]
    
    # Domain sözlüğü
    concept_lex = {
        "sigorta", "faz", "problem", "arıza", "topraklama", "izolasyon",
        "klemens", "kablo", "kontakt", "kaçak", "hat", "gaz", "trafo", 
        "direk", "ayırıcı", "pano", "elektrik", "sayaç", "kesici",
        "şalter", "priz", "toprak", "nötr", "voltaj", "amper", "akım"
    }
    
    # Corrector'ı oluştur
    corrector = BeamSearchCorrector(
        zemb_cli,  
        dokunma, 
        structures.lemma2id,
        weights=Weights(),
        thresholds=Thresholds(
            beam_width=10,
            tau_replace=0.30,
            tau_split=0.20,
            tau_merge=0.25,
            min_token_len_for_split=4
        )
    )
        
    self = BeamSearchCorrector(
        zemb_cli, 
        dokunma,
        structures.lemma2id,
        weights=Weights(),
        thresholds=Thresholds(
            beam_width=10,
            tau_replace=0.30,
            tau_split=0.20,
            tau_merge=0.25,
            min_token_len_for_split=4
        ))
    
    og_fider_olamayanlar_index = _input[_input["cause code"] != "OG Fider Açması"].index.to_list()  
    
    sayı = random.choice(og_fider_olamayanlar_index)
    tokens = _input.iloc[sayı]["Concatted"]
    cause_code = _input.iloc[sayı]["cause code"]
    
    tokenized = zemb.tokenize(tokens)

    # # Anlamlı tipleri al
    # meaningful_types = {'Word'}
    # tokens = [tr_lower(token) for token, token_type in tokenized if token_type in meaningful_types]
    
    # print(filtered_tokens)

    
    final_tokens = []
    
    for token, token_type in tokenized:
        if token_type in {'Word', 'WordWithSymbol', 'UnknownWord'}:
            # WordWithSymbol ve UnknownWord'leri ayır
            if token_type in {'WordWithSymbol', 'UnknownWord'}:
                # - ve / karakterlerine göre böl
                parts = token.replace('/', '-').split('-')
                # Boş stringleri filtrele ve ekle
                final_tokens.extend([tr_lower(p) for p in parts if p])
            else:
                # Normal Word ise direkt ekle
                final_tokens.append(tr_lower(token))
    
    tokens = final_tokens
        
    result_beam = corrector.correct(tokens, verbose=False)
    result_beam[1]
    result = [t for t in result_beam if t not in STOP_WORDS]
    result
    for node in result_beam:
        s = " ".join(node.out_tokens)
        
        print(" ".join(node.out_tokens))
    
    sys.exit()
    
    matcher = IDBasedCategoryMatcher(df, zemb, structures)
    
    predictions = evaluate_all_beams(
        result_beam, cause_code, matcher, 
        normalization='power', 
        norm_param=2,  # Kare al (2), küp için 3, daha aggressive için 4
        verbose=True
    )
    # result = matcher.predict(" ".join(result_beam[0].out_tokens), cause_code, top_k=3, min_confidence=0.01)
    
    if predictions:
        print(f"\n🎯 TAHMİNLER:")
        for i, pred in enumerate(predictions, 1):
            print(f"\n  {i}. {pred['kategori']}")
            print(f"     Güven: {pred['normalized_score_pct']:.2f}% (raw: {pred['final_score']:.2f})")
            print(f"     Ağırlıklı Güven: {pred['weighted_confidence']:.2f}%")
            print(f"     Maksimum Güven: {pred['max_confidence']:.2f}%")
            print(f"     Coverage: {pred['avg_coverage']*100:.0f}%")
            print(f"     Görülme: {pred['appearance_count']}/{len(result_beam)} beam (%{pred['appearance_ratio']*100:.0f})")
            
            # Beam detaylarından match bilgilerini çıkar
            if pred['beam_details']:
                # İlk beam'in match bilgilerini göster (veya en iyi olanı)
                best_beam = max(pred['beam_details'], key=lambda x: x['confidence'])
                
                # Phrase/Keyword matches
                if 'matches' in best_beam and best_beam['matches']:
                    print(f"     Phrase/Keyword Matches:")
                    for m in best_beam['matches'][:3]:  # İlk 3'ü göster
                        print(f"       - {m['text']} ({m['match_type']}, conf: {m['confidence']})")
                
                # Single token matches (eğer varsa)
                if 'single_token_matches' in best_beam and best_beam['single_token_matches']:
                    print(f"     Single Token Matches:")
                    for st in best_beam['single_token_matches'][:3]:  # İlk 3'ü göster
                        print(f"       - '{st['token']}' ({st['found_in_count']} yerde bulundu)")
                        if st['examples']:
                            print(f"         Örnekler: {', '.join(st['examples'][:3])}")
    else:
        print("     ❌ Eşleşme yok")
    
    
    while True:
        try:
            user_input = input("\nMetin girin: ").strip()
            
            if user_input.lower() == 'q':
                print("Program sonlandırılıyor...")
                break
            
            if not user_input:
                continue
            
            # Tokenlara ayır
            tokens = user_input.split()
            
            print(f"\nGiriş: {' '.join(tokens)}")
            
            # Düzelt
            corrected = corrector.correct(tokens, verbose=False)
            
            print(f"Düzeltilmiş: {corrected}")
            
            # Değişiklik oldu mu?
            if tokens != corrected:
                print("\nYapılan değişiklikler:")
                for i, (orig, corr) in enumerate(zip(tokens, corrected.split())):
                    if orig != corr:
                        print(f"  {i+1}. '{orig}' -> '{corr}'")
                
                # Ek tokenlar varsa (split sonucu)
                if len(corrected) > len(tokens):
                    print(f"  Eklenen tokenlar: {corrected[len(tokens):]}")
            else:
                print("(Değişiklik yapılmadı)")
                
        except KeyboardInterrupt:
            print("\n\nProgram sonlandırılıyor...")
            break
        except Exception as e:
            print(f"Hata: {e}")
            import traceback
            traceback.print_exc()
            
            
