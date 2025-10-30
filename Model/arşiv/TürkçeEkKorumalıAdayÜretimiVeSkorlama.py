# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 11:57:58 2025

@author: vural.bayrakli
"""

# -*- coding: utf-8 -*-
"""
Türkçe Ek Korumalı Aday Üretimi ve Skorlama
"""

from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import re
from collections import defaultdict
from Fonksiyonlar import *

# Türkçe ek kalıpları (genişletilebilir)
TURKISH_SUFFIXES = {
    # İsim ekleri
    'lar', 'ler', 'ları', 'leri', 'lara', 'lere', 'lardan', 'lerden',
    'ların', 'lerin', 'larla', 'lerle', 'larda', 'lerde',
    'sı', 'si', 'su', 'sü', 'sını', 'sini', 'sunu', 'sünü',
    'sına', 'sine', 'suna', 'süne', 'sında', 'sinde', 'sunda', 'sünde',
    'sından', 'sinden', 'sundan', 'sünden', 'sıyla', 'siyle', 'suyla', 'süyle',
    'ı', 'i', 'u', 'ü', 'yı', 'yi', 'yu', 'yü',
    'a', 'e', 'ya', 'ye', 'da', 'de', 'ta', 'te', 'dan', 'den', 'tan', 'ten',
    'nın', 'nin', 'nun', 'nün', 'ın', 'in', 'un', 'ün',
    'ım', 'im', 'um', 'üm', 'mız', 'miz', 'muz', 'müz',
    'ımız', 'imiz', 'umuz', 'ümüz', 'ınız', 'iniz', 'unuz', 'ünüz',
    'ları', 'leri', 'm', 'n', 'mız', 'nız',
    
    # Fiil ekleri
    'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
    'mış', 'miş', 'muş', 'müş', 'mıştı', 'mişti', 'muştu', 'müştü',
    'yor', 'iyor', 'uyor', 'üyor', 'yordu', 'iyordu', 'uyordu', 'üyordu',
    'acak', 'ecek', 'yacak', 'yecek', 'acaktı', 'ecekti',
    'malı', 'meli', 'malıydı', 'meliydi',
    'sa', 'se', 'saydı', 'seydi',
    'mak', 'mek', 'ma', 'me',
}

# Ek kombinasyonları için pattern
SUFFIX_PATTERNS = [
    r'[lnmrysş][ıiuüaeo].*',  # Genel ek pattern'i
    r'[dt][ıiuü].*',           # Geçmiş zaman
    r'yor.*',                   # Şimdiki zaman
    r'[ae]c[ae]k.*',           # Gelecek zaman
]

@dataclass
class CandidateWithScore:
    """Skorlu aday kelime"""
    word: str
    edit_distance: float
    suffix_penalty: float = 0.0
    is_original: bool = False
    has_suffix_preserved: bool = False
    
    @property
    def total_score(self) -> float:
        """Toplam skor (düşük = daha iyi)"""
        score = self.edit_distance
        score += self.suffix_penalty
        if self.is_original:
            score -= 0.5  # Orijinal kelime bonusu
        if self.has_suffix_preserved:
            score -= 0.3  # Ek korunmuş bonus
        return score


class SuffixAnalyzer:
    """Türkçe ek analizi yapan sınıf"""
    
    def __init__(self, zemb=None):
        self.zemb = zemb
        self.suffix_cache = {}
        
    def extract_suffix(self, word: str) -> Tuple[str, str]:
        """
        Kelimeyi kök ve ek olarak ayır
        Returns: (kök, ek)
        """
        if word in self.suffix_cache:
            return self.suffix_cache[word]
        
        # Zemberek'ten morfolojik analiz al (eğer varsa)
        if self.zemb:
            # Burada Zemberek'in morfolojik analizini kullanabiliriz
            # Şimdilik basit bir yaklaşım
            pass
        
        # Basit suffix detection
        word_lower = tr_lower(word)
        detected_suffix = ""
        potential_root = word_lower
        
        # En uzun eşleşen eki bul
        for length in range(min(6, len(word_lower)-2), 0, -1):
            suffix_candidate = word_lower[-length:]
            # Pattern kontrolü
            for pattern in SUFFIX_PATTERNS:
                if re.match(pattern, suffix_candidate):
                    detected_suffix = suffix_candidate
                    potential_root = word_lower[:-length]
                    break
            if detected_suffix:
                break
        
        # Bilinen eklerle kontrol
        if not detected_suffix:
            for suffix in sorted(TURKISH_SUFFIXES, key=len, reverse=True):
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                    detected_suffix = suffix
                    potential_root = word_lower[:-len(suffix)]
                    break
        
        result = (potential_root, detected_suffix)
        self.suffix_cache[word] = result
        return result
    
    def calculate_suffix_penalty(self, original: str, candidate: str) -> float:
        """
        Ek kaybı/değişimi için ceza hesapla
        """
        orig_root, orig_suffix = self.extract_suffix(original)
        cand_root, cand_suffix = self.extract_suffix(candidate)
        
        penalty = 0.0
        
        # Ek tamamen kaybolmuşsa büyük ceza
        if orig_suffix and not cand_suffix:
            penalty += 1.5
            
        # Ek değişmişse orta ceza
        elif orig_suffix and cand_suffix and orig_suffix != cand_suffix:
            # Benzer ekler mi kontrol et (örn: -ları/-leri)
            if self._are_similar_suffixes(orig_suffix, cand_suffix):
                penalty += 0.3  # Küçük ceza
            else:
                penalty += 0.8  # Büyük ceza
        
        # Orijinalde ek yokken aday'da ek varsa
        elif not orig_suffix and cand_suffix:
            penalty += 0.5  # Orta ceza
        
        return penalty
    
    def _are_similar_suffixes(self, suffix1: str, suffix2: str) -> bool:
        """İki ekin benzer olup olmadığını kontrol et (ses uyumu vb.)"""
        # Ses uyumu çiftleri
        harmony_pairs = [
            ('lar', 'ler'), ('ları', 'leri'), ('dan', 'den'), ('tan', 'ten'),
            ('ı', 'i'), ('u', 'ü'), ('a', 'e'), ('dı', 'di'), ('du', 'dü'),
            ('tı', 'ti'), ('tu', 'tü'), ('mış', 'miş'), ('muş', 'müş')
        ]
        
        for p1, p2 in harmony_pairs:
            if (suffix1 == p1 and suffix2 == p2) or (suffix1 == p2 and suffix2 == p1):
                return True
        
        # Edit distance kontrolü
        if weighted_edit_distance(suffix1, suffix2) <= 1:
            return True
        
        return False
    
    def preserve_suffix(self, root: str, original: str) -> str:
        """
        Yeni köke orijinal kelimenin ekini ekle
        """
        _, orig_suffix = self.extract_suffix(original)
        
        if orig_suffix:
            # Ses uyumunu kontrol et ve düzelt
            return self._apply_suffix_with_harmony(root, orig_suffix)
        
        return root
    
    def _apply_suffix_with_harmony(self, root: str, suffix: str) -> str:
        """
        Ses uyumuna göre eki uygula
        """
        # Basit ses uyumu kuralları (genişletilebilir)
        last_vowel = self._get_last_vowel(root)
        
        # Kalın-ince uyumu
        if last_vowel in 'aıou':  # Kalın
            suffix = suffix.replace('e', 'a').replace('i', 'ı')
        else:  # İnce
            suffix = suffix.replace('a', 'e').replace('ı', 'i')
        
        # Düz-yuvarlak uyumu
        if last_vowel in 'öü':  # Yuvarlak
            suffix = suffix.replace('ı', 'u').replace('i', 'ü')
        
        return root + suffix
    
    def _get_last_vowel(self, word: str) -> str:
        """Kelimedeki son ünlüyü bul"""
        vowels = 'aeıioöuü'
        for char in reversed(word.lower()):
            if char in vowels:
                return char
        return 'a'  # Default


def replace_candidates_with_suffix_awareness(
    token: str, 
    zemb: 'ZemberekClientIface', 
    concept_lex: Set[str],
    suffix_analyzer: Optional[SuffixAnalyzer] = None
) -> List[CandidateWithScore]:
    """
    Ek korumalı kelime değiştirme adayları
    """
    token_lower = tr_lower(token)
    
    if suffix_analyzer is None:
        suffix_analyzer = SuffixAnalyzer(zemb)
    
    # Orijinal kelimenin ek analizini yap
    orig_root, orig_suffix = suffix_analyzer.extract_suffix(token_lower)
    
    candidates = []
    seen = set()
    
    # 1. Zemberek önerileri
    suggestions = zemb.suggest(token_lower)
    
    for s in suggestions:
        s_lower = tr_lower(s)
        if s_lower not in seen:
            seen.add(s_lower)
            
            # Edit distance hesapla
            dist = weighted_edit_distance(token_lower, s_lower)
            
            # Suffix penalty hesapla
            suffix_penalty = suffix_analyzer.calculate_suffix_penalty(token_lower, s_lower)
            
            # Orijinal kelime mi?
            is_original = (s_lower == token_lower)
            
            # Ek korunmuş mu?
            _, s_suffix = suffix_analyzer.extract_suffix(s_lower)
            has_suffix_preserved = (orig_suffix and s_suffix and 
                                   suffix_analyzer._are_similar_suffixes(orig_suffix, s_suffix))
            
            candidates.append(CandidateWithScore(
                word=s_lower,
                edit_distance=dist,
                suffix_penalty=suffix_penalty,
                is_original=is_original,
                has_suffix_preserved=has_suffix_preserved
            ))
    
    # 2. Eğer orijinal kelime önerilerde yoksa, ekle
    if token_lower not in seen and zemb.analyze_valid(token_lower):
        candidates.append(CandidateWithScore(
            word=token_lower,
            edit_distance=0.0,
            suffix_penalty=0.0,
            is_original=True,
            has_suffix_preserved=True
        ))
        seen.add(token_lower)
    
    # # 3. Diacritic varyantları (ek koruyarak)
    # diacritic_variants = generate_diacritic_variants(orig_root, limit=3)
    # for variant_root in diacritic_variants:
    #     # Eki koru
    #     if orig_suffix:
    #         variant = suffix_analyzer._apply_suffix_with_harmony(variant_root, orig_suffix)
    #     else:
    #         variant = variant_root
        
    #     if variant not in seen and zemb.analyze_valid(variant):
    #         seen.add(variant)
    #         dist = weighted_edit_distance(token_lower, variant)
            
    #         candidates.append(CandidateWithScore(
    #             word=variant,
    #             edit_distance=dist,
    #             suffix_penalty=0.0,  # Ek korunduğu için ceza yok
    #             is_original=False,
    #             has_suffix_preserved=True
    #         ))
    
    # 4. Domain sözlüğünden kelimeler (ek ekleyerek)
    for w in concept_lex:
        # Hem orijinal hali hem de ekli halini dene
        variants_to_try = [w]
        
        # Eğer orijinal kelimede ek varsa, domain kelimesine de ekle
        if orig_suffix:
            w_with_suffix = suffix_analyzer._apply_suffix_with_harmony(w, orig_suffix)
            variants_to_try.append(w_with_suffix)
        
        for variant in variants_to_try:
            if variant not in seen and abs(len(variant) - len(token_lower)) <= 3:
                dist = weighted_edit_distance(variant, token_lower)
                if dist <= 2.5:
                    if zemb.analyze_valid(variant):
                        seen.add(variant)
                        suffix_penalty = suffix_analyzer.calculate_suffix_penalty(token_lower, variant)
                        
                        candidates.append(CandidateWithScore(
                            word=variant,
                            edit_distance=dist,
                            suffix_penalty=suffix_penalty,
                            is_original=False,
                            has_suffix_preserved=(variant == w_with_suffix) if orig_suffix else False
                        ))
    
    # 5. Sadece kökü değiştirip eki koru
    if orig_suffix and orig_root:
        # Kök için öneriler al
        root_suggestions = zemb.suggest(orig_root)
        for root_sugg in root_suggestions[:5]:
            # Yeni köke eski eki ekle
            new_word = suffix_analyzer._apply_suffix_with_harmony(root_sugg, orig_suffix)
            if new_word not in seen and zemb.analyze_valid(new_word):
                seen.add(new_word)
                dist = weighted_edit_distance(token_lower, new_word)
                
                candidates.append(CandidateWithScore(
                    word=new_word,
                    edit_distance=dist,
                    suffix_penalty=0.0,  # Ek korunduğu için ceza yok
                    is_original=False,
                    has_suffix_preserved=True
                ))
    
    # Total score'a göre sırala
    candidates.sort(key=lambda x: x.total_score)
    
    # En iyi adayları döndür (sadece kelime listesi olarak)
    return candidates[:15]  # Biraz daha fazla aday döndür


def generate_diacritic_variants(token: str, limit: int = 5) -> Set[str]:
    """Diacritic varyantları üret (güncellendi)"""
    variants = set()
    token_lower = tr_lower(token)
    
    # Tek karakter değişimi
    for i, ch in enumerate(token_lower):
        if ch in TR_DIACRITIC_MAP:
            alt = token_lower[:i] + TR_DIACRITIC_MAP[ch] + token_lower[i+1:]
            variants.add(alt)
            if len(variants) >= limit:
                return variants
    
    # İki karakter değişimi
    if len(variants) < limit and len(token_lower) > 1:
        for i in range(len(token_lower)-1):
            for j in range(i+1, len(token_lower)):
                if token_lower[i] in TR_DIACRITIC_MAP and token_lower[j] in TR_DIACRITIC_MAP:
                    alt = list(token_lower)
                    alt[i] = TR_DIACRITIC_MAP[token_lower[i]]
                    alt[j] = TR_DIACRITIC_MAP[token_lower[j]]
                    variants.add(''.join(alt))
                    if len(variants) >= limit:
                        return variants
    
    return variants


# TokenScorer'a eklenecek güncelleme
class ImprovedTokenScorer:
    """Ek korumalı skorlama"""
    
    def __init__(self, zemb, lm, concept_lex, weights):
        self.zemb = zemb
        self.lm = lm
        self.concept_lex = concept_lex
        self.w = weights
        self.suffix_analyzer = SuffixAnalyzer(zemb)
        
        # Yeni weight ekle
        self.w.theta_suffix = 0.40  # Suffix preservation weight
    
    def score_candidate(self, left: List[str], cand: str, right: List[str], 
                        original: str, operation: str = 'REPLACE') -> float:
        """Geliştirilmiş skorlama - ek korumalı"""
        
        s = 0.0
        
        # Temel skorlar
        s += self.w.alpha_lm * self.lm_score(left, cand, right)
        s += self.morph_ok(cand)
        s += self.domain_boost(cand, left, right)
        
        # Edit distance maliyeti
        s -= self.w.delta_edit * self.edit_cost(cand, original)
        
        # Diacritic maliyeti
        s -= self.w.eps_diacrit * self.diacritic_cost(cand, original)
        
        # Klavye maliyeti
        s -= self.w.zeta_keyboard * self.keyboard_cost(cand, original)
        
        # YENİ: Suffix preservation bonus/penalty
        suffix_score = self.calculate_suffix_score(original, cand)
        s += self.w.theta_suffix * suffix_score
        
        # İşlem bonusu
        if operation == 'SPLIT':
            s += self.w.eta_split
        elif operation == 'MERGE':
            s += 0.2
        
        return s
    
    def calculate_suffix_score(self, original: str, candidate: str) -> float:
        """
        Ek korunması skoru
        Returns: pozitif = iyi, negatif = kötü
        """
        orig_root, orig_suffix = self.suffix_analyzer.extract_suffix(original)
        cand_root, cand_suffix = self.suffix_analyzer.extract_suffix(candidate)
        
        # Orijinal kelime bonusu
        if tr_lower(original) == tr_lower(candidate):
            return 1.0
        
        # Ek korunmuşsa bonus
        if orig_suffix and cand_suffix:
            if orig_suffix == cand_suffix:
                return 0.8  # Tam eşleşme
            elif self.suffix_analyzer._are_similar_suffixes(orig_suffix, cand_suffix):
                return 0.4  # Ses uyumlu eşleşme
            else:
                return -0.3  # Farklı ek
        
        # Ek kaybolmuşsa ceza
        elif orig_suffix and not cand_suffix:
            return -1.0
        
        # Gereksiz ek eklenmişse küçük ceza
        elif not orig_suffix and cand_suffix:
            return -0.2
        
        # Her ikisinde de ek yoksa nötr
        return 0.0


# Test fonksiyonu
def test_suffix_aware_candidates():
    """Test ek korumalı aday üretimi"""
    
    # Mock Zemberek
    class MockZemberek:
        def suggest(self, word):
            suggestions = {
                'arızası': ['arıza', 'arızası', 'arızalı', 'arızasız'],
                'sigortalar': ['sigorta', 'sigortalar', 'sigortalı'],
                'elektriği': ['elektrik', 'elektriği', 'elektronik'],
            }
            return suggestions.get(word, [])
        
        def analyze_valid(self, word):
            # Basit kontrol
            return len(word) > 2
    
    zemb = MockZemberek()
    suffix_analyzer = SuffixAnalyzer(zemb_cli)
    concept_lex = {'arıza', 'sigorta', 'elektrik', 'kablo'}
    
    test_words = ['arızası', 'sigortalar', 'elektriği']
    
    for word in test_words:
        print(f"\n{'='*50}")
        print(f"Test kelime: {word}")
        print(f"{'='*50}")
        
        # Ek analizi
        root, suffix = suffix_analyzer.extract_suffix(word)
        print(f"Kök: {root}, Ek: {suffix}")
        
        # Adayları üret
        candidates = replace_candidates_with_suffix_awareness(
            word, zemb_cli, concept_lex, suffix_analyzer
        )
        
        print(f"\nAdaylar (skorlu):")
        for i, cand in enumerate(candidates[:5], 1):
            print(f"{i}. {cand.word:15s} | Skor: {cand.total_score:6.3f} | "
                  f"Edit: {cand.edit_distance:.2f} | Suffix Pen: {cand.suffix_penalty:.2f} | "
                  f"Orig: {cand.is_original} | Preserved: {cand.has_suffix_preserved}")


if __name__ == "__main__":
    # Test
    pass
    # test_suffix_aware_candidates()