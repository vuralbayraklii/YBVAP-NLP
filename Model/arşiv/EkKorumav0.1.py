# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:46:37 2025

@author: vural.bayrakli
"""

# -*- coding: utf-8 -*-
"""
Zemberek Morfoloji Tabanlı Gelişmiş Ek Yönetimi
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional, Dict
from collections import defaultdict
import re
from Fonksiyonlar import *
import zemberek_grpc.language_id_pb2 as lang_pb
import zemberek_grpc.language_id_pb2_grpc as lang_svc
import zemberek_grpc.preprocess_pb2 as pre_pb
import zemberek_grpc.preprocess_pb2_grpc as pre_svc
import zemberek_grpc.normalization_pb2 as norm_pb
import zemberek_grpc.normalization_pb2_grpc as norm_svc
import zemberek_grpc.morphology_pb2 as morph_pb
import zemberek_grpc.morphology_pb2_grpc as morph_svc
import zemberek_grpc.spellcheck_pb2 as spellcheck
import zemberek_grpc.spellcheck_pb2_grpc as spellcheck_svc

# Morfolojik etiketler ve Türkçe karşılıkları
MORPHEME_TAGS = {
    # Nominal Morphemes
    'A3sg': '',  # 3. tekil şahıs (işaretsiz)
    'A3pl': 'lar/ler',
    'Nom': '',  # Yalın hal (işaretsiz)
    'Acc': 'ı/i/u/ü/yı/yi/yu/yü',  # Belirtme hali
    'Dat': 'a/e/ya/ye',  # Yönelme hali
    'Loc': 'da/de/ta/te',  # Bulunma hali
    'Abl': 'dan/den/tan/ten',  # Ayrılma hali
    'Gen': 'ın/in/un/ün/nın/nin/nun/nün',  # Tamlayan eki
    'Ins': 'la/le/yla/yle',  # Vasıta hali
    'P1sg': 'm/ım/im/um/üm',  # 1. tekil iyelik
    'P2sg': 'n/ın/in/un/ün',  # 2. tekil iyelik
    'P3sg': 'ı/i/u/ü/sı/si/su/sü',  # 3. tekil iyelik
    'P1pl': 'mız/miz/muz/müz/ımız/imiz/umuz/ümüz',  # 1. çoğul iyelik
    'P2pl': 'nız/niz/nuz/nüz/ınız/iniz/unuz/ünüz',  # 2. çoğul iyelik
    'P3pl': 'ları/leri',  # 3. çoğul iyelik
    
    # Verbal Morphemes
    'Past': 'dı/di/du/dü/tı/ti/tu/tü',
    'Prog1': 'yor/iyor/uyor/üyor',
    'Fut': 'acak/ecek/yacak/yecek',
    'Aor': 'ar/er/ır/ir/ur/ür',
    'Opt': 'a/e/ya/ye',
    'Cond': 'sa/se',
    'Neces': 'malı/meli',
    'Pass': 'ıl/il/ul/ül/n',
    'Caus': 'dır/dir/dur/dür/tır/tir/tur/tür',
}

@dataclass
class MorphAnalysis:
    """Morfolojik analiz sonucu"""
    token: str
    lemma: str
    pos: str  # Part of Speech
    morphemes: List[str]
    surface_forms: Dict[str, str] = field(default_factory=dict)
    
    @property
    def root(self) -> str:
        """Kök kelimeyi döndür"""
        return self.lemma
    
    @property
    def suffixes(self) -> List[str]:
        """Ekleri döndür (A3sg gibi işaretsizleri hariç)"""
        return [m for m in self.morphemes if m not in ['A3sg', 'Nom', 'Pnon']]
    
    @property
    def suffix_string(self) -> str:
        """Ekleri string olarak döndür"""
        suffix_parts = []
        for morpheme, surface in self.surface_forms.items():
            if surface and morpheme not in ['Noun', 'Verb', 'Adj', 'Adv']:
                suffix_parts.append(surface)
        return ''.join(suffix_parts)


self = ZemberekMorphologyHandler(zemb)
word = "arızasi"
suggestions, _ = zemb.spell_suggest(word)

class ZemberekMorphologyHandler:
    """Zemberek morfoloji tabanlı ek yönetimi"""
    
    def __init__(self, zemberek_client):
        self.zemb = zemberek_client
        self.cache = {}
        self.lemma_cache = {}
        
    def analyze(self, word: str) -> Optional[MorphAnalysis]:
        """Kelimeyi morfolojik olarak analiz et"""
        
        if word in self.cache:
            return self.cache[word]
        
        try:
            # Zemberek'ten analiz al
            resp = self.zemb.morph.AnalyzeSentence(
                morph_pb.SentenceAnalysisRequest(input=word)
            )
            
            if resp.results and resp.results[0].best:
                best = resp.results[0].best
                
                # Morpheme'leri ve surface form'ları ayıkla
                morphemes = []
                surface_forms = {}
                
                for m in best.morphemes:
                    if m.morpheme:
                        morphemes.append(m.morpheme)
                        if m.surface:
                            surface_forms[m.morpheme] = m.surface
                
                analysis = MorphAnalysis(
                    token=word,
                    lemma=best.lemmas if best.lemmas else word,
                    pos=best.pos if best.pos else "Unknown",
                    morphemes=morphemes,
                    surface_forms=surface_forms
                )
                
                self.cache[word] = analysis
                return analysis
                
        except Exception as e:
            print(f"Morfolojik analiz hatası: {e}")
        
        return None
    
    def extract_root_and_suffix(self, word: str) -> Tuple[str, str, List[str]]:
        """
        Kelimeyi kök ve eklere ayır
        Returns: (kök, ek_string, morpheme_listesi)
        """
        analysis = self.analyze(word)
        
        if analysis and analysis.lemma != "UNK":
            return (
                analysis.root,
                analysis.suffix_string,
                analysis.suffixes
            )
        
        # Analiz başarısızsa basit yaklaşım
        return (word, "", [])
    
    def transfer_suffixes(self, source_word: str, target_root: str) -> List[str]:
        """
        Kaynak kelimenin eklerini hedef köke transfer et
        Returns: Olası varyantların listesi
        """
        source_analysis = self.analyze(source_word)
        
        if not source_analysis or not source_analysis.suffixes:
            return [target_root]
        
        variants = []
        
        # 1. Direkt ekleme (basit yaklaşım)
        simple_suffix = source_analysis.suffix_string
        if simple_suffix:
            variants.append(target_root + simple_suffix)
        
        # 2. Ses uyumlu ekleme
        harmonized = self._apply_vowel_harmony(target_root, source_analysis.suffixes)
        if harmonized and harmonized not in variants:
            variants.append(harmonized)
        
        # 3. Zemberek'e sor (en güvenilir)
        # Target root + morpheme'leri birleştirip Zemberek'ten generate et
        for variant in variants[:]:
            if self.is_valid(variant):
                continue
            # Geçersizse ses uyumu alternatifleri dene
            alternatives = self._generate_harmony_alternatives(target_root, simple_suffix)
            variants.extend(alternatives)
        
        # Geçerli olanları filtrele ve döndür
        valid_variants = [v for v in variants if self.is_valid(v)]
        
        return valid_variants if valid_variants else [target_root]
    
    def _apply_vowel_harmony(self, root: str, morphemes: List[str]) -> str:
        """Ses uyumu kurallarına göre ekleri uygula"""
        
        result = root
        last_vowel = self._get_last_vowel(root)
        
        for morpheme in morphemes:
            if morpheme in MORPHEME_TAGS:
                suffix_variants = MORPHEME_TAGS[morpheme]
                if suffix_variants:
                    # Ses uyumuna göre doğru varyantı seç
                    suffix = self._select_harmonic_variant(last_vowel, suffix_variants)
                    result += suffix
                    last_vowel = self._get_last_vowel(result)
        
        return result
    
    def _select_harmonic_variant(self, last_vowel: str, variants: str) -> str:
        """Ses uyumuna göre doğru ek varyantını seç"""
        
        options = variants.split('/')
        if len(options) == 1:
            return options[0]
        
        # Kalın-ince uyumu
        if last_vowel in 'aıou':  # Kalın
            # Kalın varyantları tercih et
            for opt in options:
                if any(v in opt for v in 'aıou'):
                    return opt
        else:  # İnce (eiöü)
            # İnce varyantları tercih et
            for opt in options:
                if any(v in opt for v in 'eiöü'):
                    return opt
        
        # Default: ilk varyant
        return options[0]
    
    def _generate_harmony_alternatives(self, root: str, suffix: str) -> List[str]:
        """Ses uyumu alternatifleri üret"""
        alternatives = []
        
        # Temel ses uyumu dönüşümleri
        harmonies = [
            ('ı', 'i'), ('u', 'ü'), ('a', 'e'), ('o', 'ö'),
            ('lar', 'ler'), ('dan', 'den'), ('tan', 'ten'),
            ('dı', 'di'), ('du', 'dü'), ('tı', 'ti'), ('tu', 'tü')
        ]
        
        current = root + suffix
        
        for h1, h2 in harmonies:
            if h1 in suffix:
                alt_suffix = suffix.replace(h1, h2)
                alternatives.append(root + alt_suffix)
            if h2 in suffix:
                alt_suffix = suffix.replace(h2, h1)
                alternatives.append(root + alt_suffix)
        
        return list(set(alternatives))
    
    def _get_last_vowel(self, word: str) -> str:
        """Kelimedeki son ünlüyü bul"""
        vowels = 'aeıioöuüAEIİOÖUÜ'
        for char in reversed(word):
            if char in vowels:
                return char.lower()
        return 'a'
    
    def is_valid(self, word: str) -> bool:
        """Kelimenin geçerli olup olmadığını kontrol et"""
        try:
            # Önce cache'e bak
            if word in self.cache:
                return self.cache[word].pos != "Unk"
            
            # Zemberek'e sor
            is_valid, _ = self.zemb.spell_check(word)
            return is_valid
        except:
            return False
    
    def correct_with_suffix_preservation(self, word: str, suggestions: List[str]) -> List[Tuple[str, float]]:
        """
        Önerileri ek koruma skoruyla birlikte döndür
        Returns: [(kelime, ek_koruma_skoru), ...]
        """
        
        # Orijinal kelimenin analizini yap
        orig_root, orig_suffix, orig_morphemes = self.extract_root_and_suffix(word)
        
        scored_suggestions = []
        
        for suggestion in suggestions:
            suggestions_iter = iter(suggestions)
            suggestion = next(suggestions_iter)
            
            score = 0.0
            
            # Öneri analizini yap
            sugg_root, sugg_suffix, sugg_morphemes = self.extract_root_and_suffix(suggestion)
            
            
            # 1. Orijinal kelime bonusu
            if suggestion.lower() == word.lower():
                score += 2.0
                scored_suggestions.append((suggestion, score))
                continue
            
            # 2. Ek tamamen korunmuşsa yüksek bonus
            if orig_suffix and sugg_suffix:
                if orig_suffix == sugg_suffix:
                    score += 1.5
                elif set(orig_morphemes) == set(sugg_morphemes):
                    score += 1.0  # Morpheme'ler aynı ama surface farklı (ses uyumu)
                elif len(set(orig_morphemes) & set(sugg_morphemes)) > 0:
                    score += 0.5  # Kısmi eşleşme
            
            # 3. Ek kaybı cezası
            if orig_suffix and not sugg_suffix:
                score -= 2.0
            
            # 4. Gereksiz ek eklenmesi
            if not orig_suffix and sugg_suffix:
                score -= 0.5
            
            # 5. Kök benzerliği bonusu
            if orig_root and sugg_root:
                # Diacritic farkı küçük ceza
                print("original root:", orig_root)
                
                if orig_root[0] !="UNK" and tr_lower(orig_root).replace('ı','i').replace('ş','s').replace('ğ','g').replace('ç','c').replace('ö','o').replace('ü','u') == \
                   tr_lower(sugg_root).replace('ı','i').replace('ş','s').replace('ğ','g').replace('ç','c').replace('ö','o').replace('ü','u'):
                    score += 0.3
            
            scored_suggestions.append((suggestion, score))
        
        # Eğer orijinal kelimede ek varsa, kökü düzeltip eki koruyarak yeni öneriler üret
        if orig_suffix and orig_root:
            for suggestion in suggestions:
                sugg_root, _, _ = self.extract_root_and_suffix(suggestion)
                
                # Sadece kök olan önerilere orijinal eki ekle
                if sugg_root == suggestion:  # Eksiz öneri
                    variants = self.transfer_suffixes(word, sugg_root)
                    for variant in variants:
                        if variant not in [s[0] for s in scored_suggestions]:
                            # Yüksek skor ver (ek korundu)
                            scored_suggestions.append((variant, 1.8))
        
        # Skora göre sırala
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_suggestions


def replace_candidates_morphology_aware(
    token: str,
    zemb_client,
    concept_lex: Set[str],
    morph_handler: Optional[ZemberekMorphologyHandler] = None
) -> List[str]:
    """
    Morfoloji tabanlı ek korumalı aday üretimi
    """
    
    if morph_handler is None:
        morph_handler = ZemberekMorphologyHandler(zemb_client)
    
    token_lower = tr_lower(token)
    
    # 1. Zemberek'ten öneriler al
    base_suggestions = zemb_client.spell_suggest(token_lower)[0]
    
    # 2. Morfoloji tabanlı skorlama yap
    scored_candidates = morph_handler.correct_with_suffix_preservation(
        token_lower, base_suggestions
    )
    
    # 3. Domain sözlüğünden kelimeler ekle (ekli halleriyle)
    orig_root, orig_suffix, _ = morph_handler.extract_root_and_suffix(token_lower)
    
    if orig_suffix:
        for domain_word in concept_lex:
            # Domain kelimesine orijinal eki ekle
            variants = morph_handler.transfer_suffixes(token_lower, domain_word)
            for variant in variants:
                if variant not in [c[0] for c in scored_candidates]:
                    # Edit distance'a göre skorla
                    dist = weighted_edit_distance(token_lower, variant)
                    if dist <= 3.0:
                        score = 1.0 - (dist / 3.0)  # Distance'ı skora çevir
                        scored_candidates.append((variant, score))
    
    # 4. Diacritic varyantları (ek koruyarak)
    if orig_root:
        diacritic_variants = generate_diacritic_variants(orig_root, limit=3)
        for variant_root in diacritic_variants:
            if orig_suffix:
                # Varyant köke orijinal eki ekle
                variants = morph_handler.transfer_suffixes(token_lower, variant_root)
                for variant in variants:
                    if variant not in [c[0] for c in scored_candidates]:
                        scored_candidates.append((variant, 0.8))
            else:
                # Eksiz varyant
                if morph_handler.is_valid(variant_root):
                    scored_candidates.append((variant_root, 0.5))
    
    # 5. Orijinal kelimeyi ekle (eğer yoksa)
    if token_lower not in [c[0] for c in scored_candidates]:
        if morph_handler.is_valid(token_lower):
            scored_candidates.insert(0, (token_lower, 2.5))  # En yüksek skor
    
    # Skora göre sırala ve sadece kelimeleri döndür
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [candidate for candidate, score in scored_candidates[:15]]


def tr_lower(s: str) -> str:
    """Türkçe kurallara uygun lowercase"""
    return s.replace('I', 'ı').replace('İ', 'i').lower()

def weighted_edit_distance(s1: str, s2: str) -> float:
    """Ağırlıklı edit distance (basit versiyon)"""
    # Implementasyonunuz...
    return abs(len(s1) - len(s2))  # Placeholder

def generate_diacritic_variants(token: str, limit: int = 5) -> Set[str]:
    """Diacritic varyantları üret"""
    TR_DIACRITIC_MAP = {
        'i':'ı', 'ı':'i', 'g':'ğ', 'ğ':'g', 's':'ş', 'ş':'s',
        'c':'ç', 'ç':'c', 'o':'ö', 'ö':'o', 'u':'ü', 'ü':'u'
    }
    
    variants = set()
    for i, ch in enumerate(token):
        if ch in TR_DIACRITIC_MAP:
            variant = token[:i] + TR_DIACRITIC_MAP[ch] + token[i+1:]
            variants.add(variant)
            if len(variants) >= limit:
                break
    return variants


# Test fonksiyonu
def test_morphology_based_correction():
    """Morfoloji tabanlı düzeltmeyi test et"""
    
    print("Zemberek Morfoloji Tabanlı Ek Koruma Testi")
    print("="*60)
    
    zemb = ZemberekClient(host="localhost", port=6789)
    morph_handler = ZemberekMorphologyHandler(zemb)
    
    test_cases = [
        ("arızasi", None),
        ("sigoralar", ["sigorta", "sigortalar", "sigortalı"]),
        ("elektrigi", ["elektrik", "elektriği", "elektronik"]),
        ("kaplomuz", ["kablo", "kablomuz", "kablonuz"]),
    ]
    
    for word, suggestions in test_cases:
        if suggestions is None:
            suggestions = zemb.spell_suggest(word)[0]
        print(f"\nTest: {word}")
        print(f"Öneriler: {suggestions}")
        
        
        root, suffix, morphemes = morph_handler.extract_root_and_suffix(word)
        print(f"Analiz: Kök='{root}', Ek='{suffix}', Morphemes={morphemes}")
        
        
        scored = morph_handler.correct_with_suffix_preservation(word, suggestions)
        print("Skorlu sonuçlar:")
        for candidate, score in scored[:5]:
            print(f"  {candidate:15s} : {score:+.2f}")


if __name__ == "__main__":
    test_morphology_based_correction()