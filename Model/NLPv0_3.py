# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 17:53:49 2025

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
from idBasedCategoryMatcher_v4 import IDBasedCategoryMatcher
import json
from config import NLPConfig
from Levenshtein import distance
import sys
from arıza_işleme_v2 import *
import random
from tqdm import tqdm
import re

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
print(config.çözüm_açıklama_to_cause_code_path)

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
        
        if sonuclar[0]["pos"] == "Verb":
            return tr_lower(sonuclar[0]["lemmas"][0])
        
        else:
            
            return tr_lower(sonuclar[0]["lemmas"][-1])


# -----------------------------
# Beam search corrector
# -----------------------------

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
    
    def safe_get_lemma(self, word):
        """Lemma al, UNK ise orijinali döndür"""
        lemma = self.zemb.get_lemma(word)
        if not lemma or lemma.lower() == "unk":
            return word
        return lemma


    def correct(self, tokens: List[str], verbose: bool = False) -> List[BeamNode]:
        """
        Token listesini düzelt - 3 paralel strateji ile
        
        NODE 1: NORMALIZE PATH
        - Her tokeni normalize et
        - Normalize edilmiş tokenin KÖKÜNÜ al
               
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
        node1 = self._create_node1_normalize_hybrid(tokens, verbose)
        
        
        all_nodes = [node1]
        
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
    
    def letters_with_spaces(self, text):
        """Harf olmayan yerlere boşluk koy"""
        # Harf olmayanları boşlukla değiştir
        result = re.sub(r'[^a-zA-Z]+', ' ', text)
        # Fazla boşlukları temizle
        return ' '.join(result.split())
    
    def only_letters_turkish(self, text):
        """Türkçe karakterler dahil sadece harfler"""
        return re.sub(r'[^a-zA-ZğüşöçıİĞÜŞÖÇ]', '', text)
        
    def _create_node1_normalize_hybrid(
        self,
        tokens: List[str],
        verbose: bool = False
    ) -> BeamNode:
        """
        ✨ ÖNERİ 3: HİBRİT YAKLAŞIM (ÖNERİLEN)
        
        Strateji:
        1. Önce token-by-token kontrol (do_not_touch)
        2. Protected olmayanları batch normalize
        3. Sonuçları birleştir
        
        Avantajlar:
        - Güvenli (token kaybı yok)
        - Hızlı (batch normalize)
        - Esnek (token sayısı değişebilir)
        """
        if verbose:
            print(f"\n🔵 NODE 1: NORMALIZE PATH (HYBRID)")
        
        out_tokens = []
        operations = ['STRATEGY:NODE1(NORMALIZE_HYBRID)']
        quality_score = 0.0
        
        # Phase 1: Protected token'leri ayır
        protected_tokens = []
        unprotected_tokens = []
        token_map = []  # (index, token, is_protected)
        
        for i, tok in enumerate(tokens):
            is_protected = False
            final_tok = tok
            
            # do_not_touch kontrolü
            if tok in self.do_not_touch:
                is_protected = True
                final_tok = tok
            elif self.kelime_düzelt(tok)["token"] in self.do_not_touch:
                is_protected = True
                final_tok = self.kelime_düzelt(tok)["token"]
            
            if is_protected:
                protected_tokens.append((i, final_tok))
                token_map.append((i, final_tok, True))
            else:
                unprotected_tokens.append((i, tok))
                token_map.append((i, tok, False))
        
        if verbose:
            print(f"   Protected: {len(protected_tokens)}")
            print(f"   Unprotected: {len(unprotected_tokens)}")
        
        # Phase 2: Unprotected token'leri batch normalize
        if unprotected_tokens:
            unprotected_text = " ".join([t[1] for t in unprotected_tokens])
            normalized_text = self.zemb.normalize(unprotected_text)
            normalized_list = normalized_text.split()
            
            # Phase 3: Normalized token'leri temizle
            normalized_clean = []
            for norm in normalized_list:
                norm_clean = self.only_letters_turkish(norm)
                if norm_clean and norm_clean != "UNK":
                    normalized_clean.append(norm_clean)
            
            # Phase 4: Token mapping
            # Eğer token sayısı değişmediyse 1-1 eşleştir
            if len(normalized_clean) == len(unprotected_tokens):
                for (orig_idx, orig_tok), norm_tok in zip(unprotected_tokens, normalized_clean):
                    # token_map'te güncelle
                    for j, (idx, tok, is_prot) in enumerate(token_map):
                        if idx == orig_idx and not is_prot:
                            token_map[j] = (idx, norm_tok, False)
                            break
            else:
                # Token sayısı değişti - hepsini normalize edilmiş haliyle kullan
                if verbose:
                    print(f"   ⚠️  Token sayısı değişti: {len(unprotected_tokens)} → {len(normalized_clean)}")
                
                # Basit strateji: Protected'ları koru, diğerlerini sırayla ekle
                norm_idx = 0
                for j, (idx, tok, is_prot) in enumerate(token_map):
                    if not is_prot and norm_idx < len(normalized_clean):
                        token_map[j] = (idx, normalized_clean[norm_idx], False)
                        norm_idx += 1
                
                # Kalan normalized token'leri sona ekle
                while norm_idx < len(normalized_clean):
                    token_map.append((len(token_map), normalized_clean[norm_idx], False))
                    norm_idx += 1
        
        # Phase 5: Final output oluştur
        out_tokens = [tok for _, tok, _ in token_map]
        
        if verbose:
            print(f"   Final: {len(out_tokens)} token")
            for i, (orig, final) in enumerate(zip(tokens, out_tokens[:len(tokens)])):
                if orig != final:
                    print(f"      [{i}] '{orig}' → '{final}'")
        
        return BeamNode(
            i=len(tokens),
            out_tokens=out_tokens,
            operations=operations,
            quality_score=quality_score
        )

    
    def kelime_düzelt(self, tok):
        # ============================================================
        # 1. LEVENSHTEIN
        # ============================================================
        best_lev_candidates = []
        closest_candidate = None
        min_distance = float('inf')
        
        sözlük = self.do_not_touch | set(self.lemma2id.keys())

        
        for w in sözlük:
            # Uzunluk farkı kontrolü
            len_diff = abs(len(w) - len(tok))
            
            # Mesafe hesapla (HER ZAMAN hesapla)
            dist = distance(w, tok)
            
            # En yakın olanı güncelle (limit olmadan)
            if dist < min_distance:
                min_distance = dist
                closest_candidate = {
                    'token': w,
                    'consumed': 1,
                    'operation': f'LEV→DICT({tok}→{w},d={dist:.1f})',
                    'quality_score': 4.0 - (dist * 0.5),
                    'dist': dist
                }
            
            # Kısa kelimeler için özel kural (sadece best_lev için)
            if len(tok) <= 2:
                if len_diff > 1:
                    continue
                max_dist = 1
            elif len(tok) <= 4:
                if len_diff > 2:
                    continue
                max_dist = 1.5
            else:
                if len_diff > 3:
                    continue
                max_dist = 2.5
            
            # Best adaylar için mesafe kontrolü
            if dist <= max_dist:
                w_lemma = self.safe_get_lemma(w)
                
                if self.arıza_içinde_v2(w_lemma):
                    quality_bonus = 5.0 - (dist * 2.0) if len(tok) <= 2 else 4.0 - (dist * 0.5)
                    
                    best_lev_candidates.append({
                        'token': w,
                        'consumed': 1,
                        'operation': f'LEV→DICT({tok}→{w},d={dist:.1f})',
                        'quality_score': quality_bonus,
                        'dist': dist
                    })

        # Döndürme mantığı
        if best_lev_candidates:
            # Best varsa ondan en iyisini döndür
            best_lev_candidates.sort(key=lambda x: (x['dist'], -x['quality_score']))
            return best_lev_candidates[0]
        else:
            # Best yoksa KESİNLİKLE en yakın olanı döndür
            return closest_candidate  # Bu asla None olmayacak (lemma2id boş değilse)
    
    
    
def process_all_data(_input, zemb, matcher, corrector, STOP_WORDS, df, structures):
    """
    ✨ IDF SKORLAMALI - Tüm input verisini işleyip sonuçları DataFrame'e kaydeder
    
    Yeni Özellikler:
    - IDF skorları gösteriliyor
    - Raw confidence (IDF ağırlıklı) eklendi
    - Method dağılımı detaylı (direct_from_kök_neden, direct_from_kategori, text_matching_with_idf)
    - Şebeke unsuru ve arama modu bilgileri
    
    Args:
        _input: Input DataFrame
        zemb: Zemberek analyzer
        matcher: IDBasedCategoryMatcher instance (IDF skorlamalı)
        corrector: Spell corrector
        STOP_WORDS: Stop words (set, list veya dict olabilir)
        df: Arıza DataFrame
        structures: ArızaStructures instance
    
    Returns:
        results_df: Sonuçları içeren DataFrame
    """
    
    # ✨ FIX: STOP_WORDS'ü set'e çevir (dict ise)
    if isinstance(STOP_WORDS, dict):
        stop_words_set = set(STOP_WORDS.keys())
        print("⚠️  STOP_WORDS dict formatında, set'e çevrildi")
    elif isinstance(STOP_WORDS, list):
        stop_words_set = set(STOP_WORDS)
        print("⚠️  STOP_WORDS list formatında, set'e çevrildi")
    else:
        stop_words_set = STOP_WORDS  # Zaten set
    
    # Sonuçları saklamak için liste
    results_list = []
    
    print(f"\n{'='*80}")
    print(f"🔄 IDF SKORLAMALI SİSTEM - TOPLU İŞLEM BAŞLIYOR")
    print(f"{'='*80}")
    print(f"📊 Toplam {len(_input)} kayıt işlenecek...")
    print(f"✨ IDF skorları her tahmin için hesaplanacak\n")
    
    # Her satırı işle
    for idx in tqdm(_input.index, desc="İşleniyor", ncols=100):
        
        row = None  # Hata durumu için
        try:
            # Veriyi al
            row = _input.loc[idx]
            tokens_raw = row["Concatted"]
            cause_code = row["cause code"]
            çözüm_açıklama = row.get("Çözüm Açıklama", None)  # ✨ .get() ile güvenli erişim
            
            # Tokenize
            tokenized = zemb.tokenize(tokens_raw)
            
            # Token işleme
            final_tokens = []
            for token, token_type in tokenized:
                if token_type in {'Word', 'WordWithSymbol', 'UnknownWord'}:
                    if token_type in {'WordWithSymbol', 'UnknownWord'}:
                        parts = token.replace('/', '-').split('-')
                        final_tokens.extend([tr_lower(p) for p in parts if p])
                    else:
                        final_tokens.append(tr_lower(token))
            
            tokens = final_tokens
            
            # Spell correction - correct() beam list döndürür
            result_beam = corrector.correct(tokens, verbose=False)
            
            # İlk beam'den tokenları al ve stop words filtrele
            if result_beam and len(result_beam) > 0:
                corrected_tokens = result_beam[0].out_tokens
                filtered_tokens = [t for t in corrected_tokens if t not in stop_words_set]
            else:
                # Düzeltme başarısızsa orijinal tokenları kullan
                filtered_tokens = [t for t in tokens if t not in stop_words_set]
            
            # ✨ YENİ: IDF skorlamalı tahmin yap (print'leri bastır)
            import sys
            import io
            
            # Standart çıktıyı geçici olarak bastır
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                result = matcher.predict(
                    user_input=" ".join(filtered_tokens), 
                    cause_code=cause_code, 
                    çözüm_açıklama=çözüm_açıklama,
                    top_k=5,
                    min_confidence=0.01
                )
            finally:
                # Standart çıktıyı geri yükle
                sys.stdout = old_stdout
            
            # Predictions'ı al
            if isinstance(result, dict) and 'predictions' in result:
                predictions = result['predictions']
            else:
                predictions = result if isinstance(result, list) else []
            
            # En iyi tahmini bul
            result_best = predictions[0]['kategori'] if predictions else "NO_MATCH"
            
            # ✨ YENİ: IDF bilgilerini ekle
            best_confidence = predictions[0].get('confidence', 0.0) if predictions else 0.0
            best_raw_confidence = predictions[0].get('raw_confidence', 0.0) if predictions else 0.0
            
            # Full predictions text oluştur
            if predictions:
                result_full_parts = []
                for i, pred in enumerate(predictions[:5], 1):
                    # Güven skorunu al
                    confidence = pred.get('confidence', 0.0)
                    raw_confidence = pred.get('raw_confidence', 0.0)
                    
                    # ✨ YENİ: IDF skorlu çıktı
                    pred_text = f"{i}. {pred['kategori']} (Conf: {confidence:.3f}"
                    
                    # Raw confidence (IDF skorlu) ekle
                    if raw_confidence > 0:
                        pred_text += f", IDF: {raw_confidence:.3f}"
                    
                    # Match summary ekle
                    if 'match_summary' in pred:
                        summary = pred['match_summary']
                        if isinstance(summary, dict):
                            exact = summary.get('exact', 0)
                            subset = summary.get('subset', 0)
                            partial = summary.get('partial', 0)
                            single = summary.get('single_token', 0)
                            total = summary.get('total', 0)
                            
                            if total > 0:
                                match_str_parts = []
                                if exact > 0:
                                    match_str_parts.append(f"E:{exact}")
                                if subset > 0:
                                    match_str_parts.append(f"S:{subset}")
                                if partial > 0:
                                    match_str_parts.append(f"P:{partial}")
                                if single > 0:
                                    match_str_parts.append(f"ST:{single}")
                                
                                if match_str_parts:
                                    pred_text += f", [{'/'.join(match_str_parts)}]"
                    
                    pred_text += ")"
                    result_full_parts.append(pred_text)
                    
                result_full = " | ".join(result_full_parts)
            else:
                result_full = "NO_MATCH"
            
            # ✨ YENİ: Ek bilgiler
            method = result.get('method', 'unknown') if isinstance(result, dict) else 'unknown'
            search_mode = result.get('search_mode', 'N/A') if isinstance(result, dict) else 'N/A'
            sebeke_unsuru = result.get('sebeke_unsuru', None) if isinstance(result, dict) else None
            
            # Sonucu listeye ekle
            results_list.append({
                'input_concatted': tokens_raw,
                'cause_code': cause_code,
                'çözüm_açıklama': çözüm_açıklama if çözüm_açıklama else "",
                'result_best': result_best,
                'result_full': result_full,
                'corrected_tokens': " ".join(filtered_tokens),
                'num_predictions': len(predictions) if predictions else 0,
                'method': method,
                'search_mode': search_mode,  # ✨ YENİ
                'sebeke_unsuru': sebeke_unsuru if sebeke_unsuru else "",  # ✨ YENİ
                'confidence': best_confidence,  # ✨ YENİ
                'raw_confidence_idf': best_raw_confidence,  # ✨ YENİ (IDF skorlu)
            })
            
        except Exception as e:
            # Hata durumunda da kaydet
            error_msg = str(e)
            
            if row is not None:
                results_list.append({
                    'input_concatted': row.get("Concatted", "ERROR"),
                    'cause_code': row.get("cause code", "ERROR"),
                    'çözüm_açıklama': row.get("Çözüm Açıklama", ""),
                    'result_best': f"ERROR: {error_msg}",
                    'result_full': f"ERROR: {error_msg}",
                    'corrected_tokens': "",
                    'num_predictions': 0,
                    'method': 'error',
                    'search_mode': 'error',
                    'sebeke_unsuru': "",
                    'confidence': 0.0,
                    'raw_confidence_idf': 0.0,
                })
            else:
                results_list.append({
                    'input_concatted': f"ERROR at index {idx}",
                    'cause_code': "ERROR",
                    'çözüm_açıklama': "",
                    'result_best': f"ERROR: {error_msg}",
                    'result_full': f"ERROR: {error_msg}",
                    'corrected_tokens': "",
                    'num_predictions': 0,
                    'method': 'error',
                    'search_mode': 'error',
                    'sebeke_unsuru': "",
                    'confidence': 0.0,
                    'raw_confidence_idf': 0.0,
                })
            
            print(f"\n⚠️ Hata (index {idx}): {error_msg}")
            print(traceback.format_exc())
    
    # DataFrame oluştur
    results_df = pd.DataFrame(results_list)
    
    # ✨ YENİ: Detaylı İstatistikler
    print(f"\n{'='*80}")
    print("📊 IDF SKORLAMALI SİSTEM - İŞLEM SONUÇLARI")
    print(f"{'='*80}")
    
    # Temel istatistikler
    print(f"\n📈 GENEL İSTATİSTİKLER:")
    print(f"   ✅ Toplam işlenen kayıt: {len(results_df)}")
    
    no_match_count = (results_df['result_best'] == 'NO_MATCH').sum()
    success_count = ((results_df['result_best'] != 'NO_MATCH') & 
                     (~results_df['result_best'].str.startswith('ERROR'))).sum()
    error_count = results_df['result_best'].str.startswith('ERROR').sum()
    
    print(f"   ✅ Başarılı eşleşme: {success_count} (%{success_count/len(results_df)*100:.1f})")
    print(f"   ❌ NO_MATCH sayısı: {no_match_count} (%{no_match_count/len(results_df)*100:.1f})")
    
    if error_count > 0:
        print(f"   ⚠️  Hatalı kayıt: {error_count} (%{error_count/len(results_df)*100:.1f})")
    
    # Method dağılımı
    if 'method' in results_df.columns:
        print(f"\n📋 METHOD DAĞILIMI:")
        method_counts = results_df['method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(results_df) * 100
            print(f"   • {method}: {count} (%{pct:.1f})")
            
            # Method'a göre alt istatistikler
            if method in ['direct_from_kök_neden', 'direct_from_kategori']:
                print(f"     → Direkt mapping (IDF kullanılmadı)")
            elif method == 'text_matching_with_idf':
                print(f"     → IDF skorlamalı text matching")
    
    # Arama modu dağılımı
    if 'search_mode' in results_df.columns:
        print(f"\n🔍 ARAMA MODU DAĞILIMI:")
        search_mode_counts = results_df['search_mode'].value_counts()
        for mode, count in search_mode_counts.items():
            if mode not in ['error', 'N/A']:
                pct = count / len(results_df) * 100
                if mode == 'filtered':
                    print(f"   • 🎯 Filtered (Şebeke unsuru bazlı): {count} (%{pct:.1f})")
                elif mode == 'global':
                    print(f"   • 🌍 Global (Tüm kategoriler): {count} (%{pct:.1f})")
    
    # IDF skorları analizi (sadece text_matching_with_idf için)
    if 'raw_confidence_idf' in results_df.columns:
        idf_results = results_df[
            (results_df['method'] == 'text_matching_with_idf') & 
            (results_df['raw_confidence_idf'] > 0)
        ]
        
        if len(idf_results) > 0:
            print(f"\n💎 IDF SKOR ANALİZİ (Text Matching için):")
            print(f"   • Ortalama IDF Skor: {idf_results['raw_confidence_idf'].mean():.3f}")
            print(f"   • Medyan IDF Skor: {idf_results['raw_confidence_idf'].median():.3f}")
            print(f"   • Min IDF Skor: {idf_results['raw_confidence_idf'].min():.3f}")
            print(f"   • Max IDF Skor: {idf_results['raw_confidence_idf'].max():.3f}")
            
            # Yüksek IDF skorlu tahminler (>2.0)
            high_idf = idf_results[idf_results['raw_confidence_idf'] > 2.0]
            if len(high_idf) > 0:
                print(f"   ⭐ Yüksek IDF (>2.0): {len(high_idf)} (%{len(high_idf)/len(idf_results)*100:.1f})")
                print(f"      → Bu tahminler çok ayırt edici kelimeler içeriyor!")
    
    # Confidence dağılımı
    if 'confidence' in results_df.columns:
        conf_results = results_df[results_df['confidence'] > 0]
        
        if len(conf_results) > 0:
            print(f"\n📊 CONFIDENCE DAĞILIMI:")
            print(f"   • Ortalama Confidence: {conf_results['confidence'].mean():.3f}")
            print(f"   • Medyan Confidence: {conf_results['confidence'].median():.3f}")
            
            # Confidence aralıkları
            high_conf = conf_results[conf_results['confidence'] >= 0.8]
            med_conf = conf_results[(conf_results['confidence'] >= 0.5) & (conf_results['confidence'] < 0.8)]
            low_conf = conf_results[conf_results['confidence'] < 0.5]
            
            print(f"   • Yüksek Güven (≥0.8): {len(high_conf)} (%{len(high_conf)/len(conf_results)*100:.1f})")
            print(f"   • Orta Güven (0.5-0.8): {len(med_conf)} (%{len(med_conf)/len(conf_results)*100:.1f})")
            print(f"   • Düşük Güven (<0.5): {len(low_conf)} (%{len(low_conf)/len(conf_results)*100:.1f})")
    
    # Şebeke unsuru dağılımı
    if 'sebeke_unsuru' in results_df.columns:
        unsur_results = results_df[results_df['sebeke_unsuru'] != ""]
        
        if len(unsur_results) > 0:
            print(f"\n🏗️ ŞEBEKE UNSURU DAĞILIMI:")
            unsur_counts = unsur_results['sebeke_unsuru'].value_counts().head(10)
            for unsur, count in unsur_counts.items():
                pct = count / len(results_df) * 100
                print(f"   • {unsur}: {count} (%{pct:.1f})")
    
    print(f"\n{'='*80}")
    print("✅ İşlem tamamlandı!")
    print(f"{'='*80}\n")
    
    return results_df


def print_result(result):
    """Sonuçları formatlanmış şekilde yazdır"""
    
    print("="*80)
    print("📊 ARIZA TESPİT SONUCU")
    print("="*80)
    
    # Temel bilgiler
    print(f"\n📝 GİRDİ BİLGİLERİ:")
    print(f"   Metin: {result['input'][:100]}..." if len(result['input']) > 100 else f"   Metin: {result['input']}")
    print(f"   Sebep Kodu: {result['cause_code']}")
    print(f"   Şebeke Unsuru: {result['sebeke_unsuru']}")
    print(f"   Arama Modu: {result['search_mode']}")
    
    # Kelime sayıları
    print(f"\n📈 KELİME İSTATİSTİKLERİ:")
    print(f"   Toplam Kelime: {len(result['input_words'])}")
    print(f"   Benzersiz Kelime: {len(set(result['input_words']))}")
    
    # Eşleşme istatistikleri
    print(f"\n🎯 EŞLEŞME İSTATİSTİKLERİ:")
    print(f"   Tam Eşleşme: {result['total_exact_matches']}")
    print(f"   Alt Küme Eşleşmesi: {result['total_subset_matches']}")
    print(f"   Kısmi Eşleşme: {result['total_partial_matches']}")
    print(f"   Tek Token Eşleşmesi: {result['total_single_token_matches']}")
    
    # Tahminler
    print(f"\n🔍 TAHMİNLER (İlk 5):")
    print("-"*80)
    
    for i, pred in enumerate(result['predictions'][:5], 1):
        # Başlık satırı
        print(f"\n{i}. {pred['kategori']}")
        print(f"   {'▓'*int(pred['confidence_pct']/2)}{'░'*(50-int(pred['confidence_pct']/2))} %{pred['confidence_pct']:.2f}")
        
        # Detaylar
        print(f"   • Güven Skoru: {pred['confidence']:.4f}")
        print(f"   • Ham Skor: {pred['raw_score']:.2f}")
        print(f"   • Kapsam: %{pred['coverage']*100:.0f}")
        print(f"   • Eşleşme Sayısı: {pred['match_count']}")
        
        # Eşleşme tipleri
        if pred['match_types']:
            types_str = ", ".join([f"{k}: {v}" for k, v in pred['match_types'].items()])
            print(f"   • Eşleşme Tipleri: {types_str}")
        
        # Örnekler
        if pred.get('matches'):
            print(f"   • Cümle Eşleşmeleri:")
            for match in pred['matches'][:3]:  # İlk 3 örnek
                print(f"      - '{match['text']}' ({match['match_type']}, güven: {match['confidence']})")
        
        if pred.get('single_token_matches'):
            print(f"   • Kelime Eşleşmeleri:")
            for token_match in pred['single_token_matches'][:2]:  # İlk 2 token
                examples = ", ".join([f"'{ex}'" for ex in token_match['examples'][:2]])
                print(f"      - '{token_match['token']}' (güven: {token_match['confidence']}, örnekler: {examples})")
    
    print("\n" + "="*80)


def Test(_input, zemb, matcher, corrector, STOP_WORDS, df, structures, idx):
    """
    Tüm input verisini işleyip sonuçları DataFrame'e kaydeder
    
    Args:
        _input: Input DataFrame
        zemb: Zemberek analyzer
        corrector: Spell corrector
        STOP_WORDS: Stop words (set, list veya dict olabilir)
        df: Arıza DataFrame
        structures: ArızaStructures instance
    
    Returns:
        results_df: Sonuçları içeren DataFrame
    """
    
    # ✨ FIX: STOP_WORDS'ü set'e çevir (dict ise)
    if isinstance(STOP_WORDS, dict):
        stop_words_set = set(STOP_WORDS.keys())
        print("⚠️  STOP_WORDS dict formatında, set'e çevrildi")
    elif isinstance(STOP_WORDS, list):
        stop_words_set = set(STOP_WORDS)
        print("⚠️  STOP_WORDS list formatında, set'e çevrildi")
    else:
        stop_words_set = STOP_WORDS  # Zaten set
    
    # Sonuçları saklamak için liste
    results_list = []
    
    # # OG Fider olmayanları filtrele
    # og_fider_olamayanlar = _input[_input["cause code"] != "OG Fider Açması"].copy()
    
    # # İndeksi sıfırla (tekrarlanan indeks değerlerini önlemek için)
    # og_fider_olamayanlar = og_fider_olamayanlar.reset_index(drop=True)
        
    # tqdm_iter = iter(tqdm(_input.index, desc="İşleniyor"))
    # idx = next(tqdm_iter)
    
    row = None  # Hata durumu için
    try:
        # Veriyi al
        # idx = next(tqdm_iter)
        
        row = _input.loc[idx]
        tokens_raw = row["Concatted"]
        # tokens_raw = "BRANSMAN ARIZASI-SALIHLI	Durasilli	ATATURK	43	ACILIYET BELIRTTI.	DURASILLI TR-4_;AO - Enerji Kesintili;AG;Klemens Arızası;AG Klemens yenilendi;;BRANŞMAN ARIZASI-SALİHLİ	Durasıllı	ATATÜRK	43	ACILIYET BELIRTTI.	DURASILLI TR-4_45-78-M01146_953261945_953261945_ 		06-02-2025 15:43:07		Açık	Bağlı Değil	 Enerji Gidip-Geliyor"
        cause_code = row["cause code"]
        çözüm_açıklama = row.get("Çözüm Açıklama", None)  # ✨ .get() ile güvenli erişim
        
        # Tokenize
        tokenized = zemb.tokenize(tokens_raw)
        
        # Token işleme
        final_tokens = []
        for token, token_type in tokenized:
            if token_type in {'Word', 'WordWithSymbol', 'UnknownWord'}:
                if token_type in {'WordWithSymbol', 'UnknownWord'}:
                    parts = token.replace('/', '-').split('-')
                    final_tokens.extend([tr_lower(p) for p in parts if p])
                else:
                    final_tokens.append(tr_lower(token))
        
        tokens = final_tokens
        
        # Spell correction - correct() beam list döndürür
        result_beam = corrector.correct(tokens, verbose=False)
        
        # İlk beam'den tokenları al ve stop words filtrele
        if result_beam and len(result_beam) > 0:
            corrected_tokens = result_beam[0].out_tokens
            # ✨ FIX: set kullan
            filtered_tokens = [t for t in corrected_tokens if t not in stop_words_set]
        else:
            # Düzeltme başarısızsa orijinal tokenları kullan
            # ✨ FIX: set kullan
            filtered_tokens = [t for t in tokens if t not in stop_words_set]
        
        # Tahmin yap
        result = matcher.predict(
            user_input=" ".join(filtered_tokens), 
            cause_code=cause_code, 
            çözüm_açıklama=çözüm_açıklama,
            top_k=5,
            min_confidence=0.01
        )
        
        # Predictions'ı al
        if isinstance(result, dict) and 'predictions' in result:
            predictions = result['predictions']
        else:
            predictions = result if isinstance(result, list) else []
        
        # En iyi tahmini bul
        result_best = predictions[0]['kategori'] if predictions else "NO_MATCH"
        
        # Full predictions text oluştur
        if predictions:
            result_full_parts = []
            for i, pred in enumerate(predictions[:5], 1):
                # ✨ FIX: Mevcut alan adlarını kontrol et
                confidence_key = 'confidence' if 'confidence' in pred else 'confidence_pct'
                
                # Güven skorunu al
                confidence = pred.get(confidence_key, 0.0)
                
                pred_text = f"{i}. {pred['kategori']} (Güven: {confidence:.2f}"
                
                # Opsiyonel alanları ekle
                if 'coverage' in pred:
                    coverage_val = pred['coverage']
                    if coverage_val < 1:  # Oran olarak verilmişse
                        coverage_val *= 100
                    pred_text += f", Kapsam: {coverage_val:.0f}%"
                elif 'avg_coverage' in pred:
                    coverage_val = pred['avg_coverage']
                    if coverage_val < 1:
                        coverage_val *= 100
                    pred_text += f", Kapsam: {coverage_val:.0f}%"
                
                if 'match_count' in pred:
                    pred_text += f", Eşleşme: {pred['match_count']}"
                
                # Match summary ekle
                if 'match_summary' in pred:
                    summary = pred['match_summary']
                    if isinstance(summary, dict):
                        total = summary.get('total', 0)
                        if total > 0:
                            pred_text += f", Toplam: {total}"
                
                pred_text += ")"
                result_full_parts.append(pred_text)
                
            result_full = " | ".join(result_full_parts)
        else:
            result_full = "NO_MATCH"
        
        # Sonucu listeye ekle
        results_list.append({
            'input_concatted': tokens_raw,
            'cause_code': cause_code,
            'çözüm_açıklama': çözüm_açıklama if çözüm_açıklama else "",
            'result_best': result_best,
            'result_full': result_full,
            'corrected_tokens': " ".join(filtered_tokens),
            'num_predictions': len(predictions) if predictions else 0,
            'method': result.get('method', 'unknown') if isinstance(result, dict) else 'unknown'
        })
        
    except Exception as e:
        # Hata durumunda da kaydet
        error_msg = str(e)
        
        if row is not None:
            results_list.append({
                'input_concatted': row.get("Concatted", "ERROR"),
                'cause_code': row.get("cause code", "ERROR"),
                'çözüm_açıklama': row.get("Çözüm Açıklama", ""),
                'result_best': f"ERROR: {error_msg}",
                'result_full': f"ERROR: {error_msg}",
                'corrected_tokens': "",
                'num_predictions': 0,
                'method': 'error'
            })
        else:
            results_list.append({
                'input_concatted': f"ERROR at index {idx}",
                'cause_code': "ERROR",
                'çözüm_açıklama': "",
                'result_best': f"ERROR: {error_msg}",
                'result_full': f"ERROR: {error_msg}",
                'corrected_tokens': "",
                'num_predictions': 0,
                'method': 'error'
            })
        
        print(f"\n⚠️ Hata (index {idx}): {error_msg}")
        
        # Debug için daha fazla bilgi
        import traceback
        print(traceback.format_exc())

    # DataFrame oluştur
    results_df = pd.DataFrame(results_list)
    
    # Özet istatistikler
    print(f"\n{'='*60}")
    print("📊 İŞLEM SONUÇLARI")
    print(f"{'='*60}")
    print(f"✅ Toplam işlenen kayıt: {len(results_df)}")
    print(f"❌ NO_MATCH sayısı: {(results_df['result_best'] == 'NO_MATCH').sum()}")
    print(f"✅ Başarılı eşleşme: {(results_df['result_best'] != 'NO_MATCH').sum()}")
    
    # Hata sayısını göster
    error_count = results_df['result_best'].str.startswith('ERROR').sum()
    if error_count > 0:
        print(f"⚠️  Hatalı kayıt: {error_count}")
    
    # Method dağılımı (eğer varsa)
    if 'method' in results_df.columns:
        print(f"\n📋 Method Dağılımı:")
        method_counts = results_df['method'].value_counts()
        for method, count in method_counts.items():
            print(f"   {method}: {count}")
    
    return results_df
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
    
    çözüm_açıklama_to_cause_code = pd.read_excel(config.çözüm_açıklama_to_cause_code_path)    
    
    çözüm_açıklama_info = pd.read_excel(config.çözüm_açıklama_info_path)
    
    structures = build_morphosemantic_structures_v3(
        df,
        zemb,
        config.cause_code_şebeke_unsuru_path,
        dokunma,
        çözüm_açıklama_info,
        çözüm_açıklama_to_cause_code
    )

    _input = pd.read_excel(config.input_concatted_path)
    
    sayilar = random.choices(range(1, len(_input)), k=250)
    sayilar = sorted(sayilar)
    
    input_sample = _input.iloc[sayilar]
    input_sample.reset_index(drop=True, inplace = True)
    
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
    
    matcher = IDBasedCategoryMatcher(df, zemb, structures)
    
    Test(_input, zemb, corrector, matcher)
    
    final_df = process_all_data(
        _input=_input,
        zemb=zemb,
        matcher=matcher,
        corrector=corrector,
        STOP_WORDS=STOP_WORDS,
        df=df,
        structures=structures
    )

    # CSV olarak kaydet
    final_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    print("\n💾 Sonuçlar 'prediction_results.csv' dosyasına kaydedildi.")

    # Excel olarak kaydet (opsiyonel)
    final_df.to_excel('input_concatted_prediction_results4.xlsx', index=False, engine='openpyxl')
    print("💾 Sonuçlar 'prediction_results.xlsx' dosyasına kaydedildi.")
            
