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
from idBasedCategoryMatcher import IDBasedCategoryMatcher
import json
from config import NLPConfig
from Levenshtein import distance
import sys
from arıza_işleme import *
import random
from tqdm import tqdm
import re
from datetime import datetime
import traceback
from run import process_all_data

# -----------------------------
# Değişkenler
# -----------------------------
config_path = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\config\config.json"

output_dir = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\Vural\TextMiningwithGit\YBVAP-NLP\output"
os.makedirs(output_dir, exist_ok=True)

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
        
    def tokenize_with_types(self, text: str) -> List[Tuple[str, str]]:
        """Metni tokenlara ayır ve türlerini dönder"""
        try:
            tokens_with_types = self.zemberek.tokenize(text)

            return tokens_with_types if tokens_with_types else [(t, "Unknown") for t in text.split()]
        except:
            return [(t, "Unknown") for t in text.split()]
        
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

    def _process_input(self, inp):
       
        # Tokenize
        tokenized = self.zemb.tokenize_with_types(inp)
        
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
        result_beam = self.correct(tokens, verbose=False)
        
        # İlk beam'den tokenları al ve stop words filtrele
        if result_beam and len(result_beam) > 0:
            corrected_tokens = result_beam[0].out_tokens
            filtered_tokens = [t for t in corrected_tokens if t not in STOP_WORDS]
        else:
            # Düzeltme başarısızsa orijinal tokenları kullan
            filtered_tokens = [t for t in tokens if t not in STOP_WORDS]
        
        return filtered_tokens
    
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
    
    çözüm_açıklama_info = pd.read_excel(config.çözüm_açıklama_info_path, sheet_name="Çözüm_Açıklama_Şebeke_Unsuru")
    cause_code_info = pd.read_excel(config.çözüm_açıklama_info_path, sheet_name="cause_code_Şebeke_Unsuru")
    
    structures = build_morphosemantic_structures_v3(
        df,
        zemb,
        config.cause_code_şebeke_unsuru_path,
        dokunma,
        çözüm_açıklama_info,
        çözüm_açıklama_to_cause_code,
        cause_code_info
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
        
    final_df = process_all_data(
        _input=_input,
        zemb=zemb,
        matcher=matcher,
        corrector=corrector,
        STOP_WORDS=STOP_WORDS,
        df=df,
        structures=structures
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(output_dir, f"prediction_results_{timestamp}.xlsx")

    # Excel olarak kaydet (opsiyonel)
    final_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"💾 Sonuçlar '{output_path}' dosyasına kaydedildi.")

