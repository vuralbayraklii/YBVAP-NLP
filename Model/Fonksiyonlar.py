# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 12:12:20 2025

@author: vural.bayrakli
"""
from typing import List, Tuple, Set, Optional, Dict
import unicodedata
import os
import numpy as np
from collections import defaultdict
import pandas as pd
import string
from dataclasses import dataclass, field
import random

# -----------------------------
# Utilities
# -----------------------------

TR_DIACRITIC_MAP = {
    'i':'ı', 'ı':'i', 'g':'ğ', 'ğ':'g', 's':'ş', 'ş':'s',
    'c':'ç', 'ç':'c', 'o':'ö', 'ö':'o', 'u':'ü', 'ü':'u',
    'I':'İ', 'İ':'I'  # Büyük harf desteği
}


# Klavye komşuluğu (Q klavye)
KEY_NEIGHBORS = {
    'q':['w','a'], 'w':['q','e','s'], 'e':['w','r','d'], 'r':['e','t','f'], 
    't':['r','y','g'], 'y':['t','u','h'], 'u':['y','ı','j'], 'ı':['u','o','k'],
    'o':['ı','p','l'], 'p':['o','ö','ş'], 'ö':['p','ç'], 'ç':['ö'],
    
    'a':['q','s','z'], 's':['a','w','d','x'], 'd':['s','e','f','c'], 
    'f':['d','r','g','v'], 'g':['f','t','ğ','b'], 'ğ':['g','h','n'], 
    'h':['ğ','y','j','m'], 'j':['h','u','k'], 'k':['j','ı','l'],
    'l':['k','o','ş'], 'ş':['l','p','i'],
    
    'z':['a','x'], 'x':['z','s','c'], 'c':['x','d','v'], 'v':['c','f','b'],
    'b':['v','g','n'], 'n':['b','ğ','m'], 'm':['n','h'],
    
    'ü':['i','p'], 'i':['ü','ş']
}

def tr_lower(s: str) -> str:
    """Türkçe kurallara uygun lowercase"""
    return s.replace('I', 'ı').replace('İ', 'i').lower()

def normalize_text(s: str) -> str:
    """Unicode normalizasyonu ve lowercase"""
    # s = unicodedata.normalize('NFC', s)
    nfd = unicodedata.normalize('NFD', s)
    # Accent işaretlerini filtrele
    return tr_lower(''.join(char for char in nfd if unicodedata.category(char) != 'Mn'))

    # return tr_lower(s)

def is_diacritic_pair(a: str, b: str) -> bool:
    """İki karakterin diacritic çifti olup olmadığını kontrol et"""
    return (a in TR_DIACRITIC_MAP and TR_DIACRITIC_MAP[a] == b) or \
           (b in TR_DIACRITIC_MAP and TR_DIACRITIC_MAP[b] == a)

def diacritic_penalty(a: str, b: str) -> float:
    """Diacritic farklılıklarına ceza"""
    n = min(len(a), len(b))
    count = 0
    for i in range(n):
        if i < len(a) and i < len(b):
            if a[i] != b[i] and is_diacritic_pair(a[i], b[i]):
                count += 1
    return float(count) * 0.5  # Ceza azaltıldı

def sub_cost(a: str, b: str) -> float:
    """Karakter değiştirme maliyeti"""
    if a == b:
        return 0.0
    if is_diacritic_pair(a, b):
        return 0.2  # Diacritic hatası düşük maliyet
    
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower in KEY_NEIGHBORS and b_lower in KEY_NEIGHBORS[a_lower]:
        return 0.4  # Klavye komşuluğu orta maliyet
    return 1.0

def weighted_edit_distance(s: str, t: str) -> float:
    """Ağırlıklı edit distance"""
    s = normalize_text(s)
    t = normalize_text(t)
    m, n = len(s), len(t)
    
    if m == 0: return float(n)
    if n == 0: return float(m)
    
    dp = [[0.0]*(n+1) for _ in range(m+1)]
    
    for i in range(1, m+1): 
        dp[i][0] = dp[i-1][0] + 1.0
    for j in range(1, n+1): 
        dp[0][j] = dp[0][j-1] + 1.0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j] + 1.0,                         # deletion
                dp[i][j-1] + 1.0,                         # insertion
                dp[i-1][j-1] + sub_cost(s[i-1], t[j-1])  # substitution
            )
    
    return dp[m][n]

def window(tokens: List[str], i: int, k: int) -> Tuple[List[str], List[str]]:
    """Token etrafındaki pencereyi al"""
    left = tokens[max(0, i-k): i]
    right = tokens[i+1: min(len(tokens), i+1+k)]
    return left, right



# -----------------------------
# Candidate generation
# -----------------------------
def generate_diacritic_variants(token: str, zemb, limit: int = 5) -> Set[str]:
    """Diacritic varyantları üret"""
    variants = set()
    # token = "yasandigini"
    token_lower = tr_lower(token)
    
    # Tek karakter değişimi
    for i, ch in enumerate(token_lower):
        # e_iter = iter(enumerate(token_lower))
        # i, ch = next(e_iter)
        if ch in TR_DIACRITIC_MAP:
            alt = token_lower[:i] + TR_DIACRITIC_MAP[ch] + token_lower[i+1:]
            # if zemb.analyze_valid(alt):
            variants.add(alt)
            # variants.update(zemb.suggest(alt)[0] if len(zemb.suggest(alt)) > 0 else None)
    
    # İki karakter değişimi (limit dahilinde)
    if len(variants) < limit:
        for i in range(len(token_lower)):
            # i_iter = iter(range(len(token_lower)))
            # i = next(i_iter)
            for j in range(i+1, len(token_lower)):
                # j_iter = iter(range(i+1, len(token_lower)))
                # j = next(j_iter)
                if token_lower[i] in TR_DIACRITIC_MAP and token_lower[j] in TR_DIACRITIC_MAP:
                    alt = list(token_lower)
                    alt[i] = TR_DIACRITIC_MAP[token_lower[i]]
                    alt[j] = TR_DIACRITIC_MAP[token_lower[j]]
                    
                    # if zemb.analyze_valid(''.join(alt)):
                        # variants.add(alt)
                    variants.add(''.join(alt))
                    if len(variants) >= limit:
                        break
                    # variants.update(zemb.suggest(''.join(alt))[0] if len(zemb.suggest(''.join(alt))[0]) else None)
    
    return variants

def split_candidates(token: str, zemb, lemma2id, min_len: int = 2) -> List[Tuple[str, str]]:
    """Kelimeyi ikiye bölme adayları"""
    
    def arızalar_içinde(t):
        return t in lemma2id
    
    def işlem(t):
        
        norm1 = zemb.normalize(t)
        
        if arızalar_içinde(norm1) or arızalar_içinde(zemb.get_lemma(norm1)) :                
            return norm1
        
        # t1 için alternatif öneriler
        suggests = zemb.suggest(t)
        if suggests:
            for s in suggests:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s
        
        # norm1 için alternatif öneriler
        suggests = zemb.suggest(norm1)
        if suggests:
            for s in suggests:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s                              
        return None
    
    cands = []
    token_lower = tr_lower(token)

    for cut in range(min_len, len(token_lower) - min_len + 1):
        # cut_iter = iter(range(min_len, len(token_lower) - min_len + 1))
        # cut = next(cut_iter)
        
        t1, t2 = token_lower[:cut], token_lower[cut:]
        
        if arızalar_içinde(t1) and arızalar_içinde(t2):
            # İlk hali her zaman ekle
            cands = []
            cands.append((t1, t2))
            
            return cands
        
        if zemb.analyze_valid(t1) and zemb.analyze_valid(t2):
            
            cands.append((t1, t2))
            
            return cands
            
        if not arızalar_içinde(t1) and arızalar_içinde(t2):
            kelime = işlem(t1)
            
            if kelime:
                cands.append((kelime, t2))
            

        if arızalar_içinde(t1) and not arızalar_içinde(t2):
            kelime = işlem(t2)
            
            if kelime:
                cands.append((t1, kelime))
                
                
        if not arızalar_içinde(t1) and not arızalar_içinde(t2):
            
            kelime1 = işlem(t1)
            kelime2 = işlem(t2)
            
            if kelime1 and kelime2:
                cands.append((kelime1, kelime2))


    cands = list(set(cands))  # Tekrarları kaldır
    cands.sort(key=lambda x: len(x[0]) * len(x[1]), reverse=True)  # Dengeli bölmeleri tercih et
    
    return cands # En fazla 5 aday

def split_candidates_v2(token: str, zemb, lemma2id, min_len: int = 2) -> List[Tuple[str, str]]:
    """Kelimeyi ikiye bölme adayları"""
    
    def arızalar_içinde(t):
        return t in lemma2id
    
    def işlem(t):
        
        norm1 = zemb.normalize(t)
        
        if arızalar_içinde(norm1) or arızalar_içinde(zemb.get_lemma(norm1)) :                
            return norm1
        
        # t1 için alternatif öneriler
        suggests1 = zemb.suggest(t)
        if suggests1:
            for s in suggests1:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s
        
        # norm1 için alternatif öneriler
        suggests2 = zemb.suggest(norm1)
        if suggests2:
            for s in suggests2:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s
             
        suggests = suggests1 + suggests2
        if zemb.analyze_valid(norm1):
            suggests.append(norm1)
        
        res = random.choice(suggests) if len(suggests) > 0 else None
        
        return res
    
    def try_splits(check_func, process_func=None):
        """Belirli bir koşula göre split'leri kontrol et"""
        results = []
        for cut in range(min_len, len(token_lower) - min_len + 1):
            t1, t2 = token_lower[:cut], token_lower[cut:]
            
            if check_func(t1, t2):
                if process_func:
                    result = process_func(t1, t2)
                    if result:
                        results.append(result)
                else:
                    results.append((t1, t2))
        return results
    
    cands = []
    token_lower = tr_lower(token)
    
    # Öncelik 1: Her ikisi de arızalar içinde
    cands = try_splits(
        lambda t1, t2: arızalar_içinde(t1) and arızalar_içinde(t2)
    )
    if cands:
        return cands
    
    # Öncelik 2: Her ikisi de valid
    cands = try_splits(
        lambda t1, t2: zemb.analyze_valid(t1) and zemb.analyze_valid(t2)
    )
    if cands:
        return cands
    
    # Öncelik 3: Diğer durumlar
    def process_others(t1, t2):
        if not zemb.analyze_valid(t1) and zemb.analyze_valid(t2):
            kelime = işlem(t1)
            if kelime:                                           
                return (kelime, t2)
            
        elif zemb.analyze_valid(t1) and not zemb.analyze_valid(t2):
            kelime = işlem(t2)
            if kelime:
                return (t1, kelime)
        elif not zemb.analyze_valid(t1) and not zemb.analyze_valid(t2):
            kelime1 = işlem(t1)
            kelime2 = işlem(t2)
            if kelime1 and kelime2:
                return (kelime1, kelime2)
        return None
    
    cands = try_splits(
        lambda t1, t2: True,  # Hepsini kontrol et
        process_others
    )
    
    return random.choice(cands) if cands else []

def merge_candidates(t1: str, t2: str, zemb, concept_lex: Set[str]) -> List[str]:
    """İki kelimeyi birleştirme adayları"""
    merged = tr_lower(t1 + t2)
    cands = set([merged])
    
    # Diacritic varyantları
    # cands |= generate_diacritic_variants(merged, zemb, limit=3)
    
    # Domain sözlüğünden yakın kelimeler
    for w in concept_lex:
        if abs(len(w) - len(merged)) <= 2:
            dist = weighted_edit_distance(w, merged)
            if dist <= 2.0:
                cands.add(w)
    
    # Zemberek önerileri
    for s in zemb.suggest(merged)[:5]:
        cands.add(tr_lower(s))
    
    # Sadece geçerli olanları döndür
    valid_cands = [w for w in cands if zemb.analyze_valid(w)]
    
    # Edit distance'a göre sırala
    valid_cands.sort(key=lambda x: weighted_edit_distance(x, merged))
    
    return valid_cands[:5]


def replace_candidates(token: str, zemb, lemma2id) -> List[str]:
    """Kelime değiştirme adayları"""
    
    def arızalar_içinde(t):
        return t in lemma2id
    
    def işlem(t):
        norm1 = zemb.normalize(t)
        if norm1 and norm1 != t:
            if arızalar_içinde(norm1) or arızalar_içinde(zemb.get_lemma(norm1)) :                
                return norm1
        
        # t1 için alternatif öneriler
        suggests = zemb.suggest(t)
        if suggests:
            for s in suggests:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s
                    
        return None
    
    token_lower = tr_lower(token)
    
    base = set()
    
    # Diacritic varyantları
    base |= generate_diacritic_variants(token_lower, zemb, limit=5)
    
    # Zemberek önerileri
    if len(base) > 0:
        for b in list(base):
            
            kelime = işlem(b)
            
            if kelime:
                base.add(tr_lower(kelime))
            
            else:
                öneriler =  zemb.suggest(b)
                if len(öneriler) > 0:
                    for s in öneriler:
                        if arızalar_içinde(s):
                            base.add(tr_lower(s))
                        else:
                            kelime = işlem(s)
                            if kelime:
                                
                                base.add(tr_lower(kelime))
    
    
    öneriler =  zemb.suggest(token_lower)
    if len(öneriler) > 0:
        for s in öneriler:
            if arızalar_içinde(s):
                base.add(tr_lower(s))
            else:
                kelime = işlem(s)
                if kelime:

                    base.add(tr_lower(kelime))
            # cost = weighted_edit_distance(token_lower, s)
            # if cost <= 2:
            #     base.add(tr_lower(s))

    
    # Domain sözlüğünden yakın kelimeler
    for w in lemma2id:
        if abs(len(w) - len(token_lower)) <= 2:
            dist = weighted_edit_distance(w, token_lower)
            if dist <= 2:
                base.add(w)
                
    # Sadece geçerli olanları al
    valid = [w for w in base if zemb.analyze_valid(w)]
    
    # Edit distance'a göre sırala ve en iyileri döndür
    valid.sort(key=lambda x: weighted_edit_distance(x, token_lower))
    
    return valid[:10]  # En fazla 10 aday

def replace_candidates_v2(token: str, zemb, lemma2id) -> List[str]:
    """Kelime değiştirme adayları"""
    
    def arızalar_içinde(t):
        return t in lemma2id
    
    def işlem(t):
        
        norm1 = zemb.normalize(t)
        
        if arızalar_içinde(norm1) or arızalar_içinde(zemb.get_lemma(norm1)) :                
            return norm1
        
        # t1 için alternatif öneriler
        suggests1 = zemb.suggest(t)
        if suggests1:
            for s in suggests1:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s
        
        # norm1 için alternatif öneriler
        suggests2 = zemb.suggest(norm1)
        if suggests2:
            for s in suggests2:
                if s != t:
                    if arızalar_içinde(s) or arızalar_içinde(zemb.get_lemma(s)) :
                        return s
             
        suggests = suggests1 + suggests2
        
        res = random.choice(suggests) if len(suggests) > 0 else None
        
        return res
    
    token_lower = tr_lower(token)
    
    base = set()
    
    # Diacritic varyantları
    base |= generate_diacritic_variants(token_lower, zemb, limit=5)
    
    # Zemberek önerileri
    if len(base) > 0:
        for b in list(base):
            
            kelime = işlem(b)
            
            if kelime:
                base.add(tr_lower(kelime))
            
            else:
                öneriler =  zemb.suggest(b)
                if len(öneriler) > 0:
                    for s in öneriler:
                        if arızalar_içinde(s):
                            base.add(tr_lower(s))
                        else:
                            kelime = işlem(s)
                            if kelime:
                                
                                base.add(tr_lower(kelime))
    
    
    öneriler =  zemb.suggest(token_lower)
    if len(öneriler) > 0:
        for s in öneriler:
            if arızalar_içinde(s):
                base.add(tr_lower(s))
            else:
                kelime = işlem(s)
                if kelime:

                    base.add(tr_lower(kelime))
            # cost = weighted_edit_distance(token_lower, s)
            # if cost <= 2:
            #     base.add(tr_lower(s))

    
    # Domain sözlüğünden yakın kelimeler
    for w in lemma2id:
        if abs(len(w) - len(token_lower)) <= 2:
            dist = weighted_edit_distance(w, token_lower)
            if dist <= 2:
                base.add(w)
                
    # Sadece geçerli olanları al
    valid = [w for w in base if zemb.analyze_valid(w)]
    
    # Edit distance'a göre sırala ve en iyileri döndür
    valid.sort(key=lambda x: weighted_edit_distance(x, token_lower))
    
    return valid[:3]  # En fazla 10 aday

def replace_candidates_v3(token: str, zemb, lemma2id) -> List[str]:
    """
    Kelime değiştirme adayları - TÜM adayları topla, sözlüktekiler önce
    
    Args:
        token: Düzeltilecek kelime
        zemb: Zemberek client
        lemma2id: Arıza sözlüğü
    
    Returns:
        Aday kelime listesi (sözlüktekiler önce, sonra diğerleri, edit distance'a göre sıralı)
    """
    
    def arızalar_içinde(t):
        """Token veya lemması sözlükte mi?"""
        if t in lemma2id:
            return True
        lemma = zemb.get_lemma(t)
        if lemma and lemma in lemma2id:
            return True
        return False
    
    def işle_ve_ekle(t, candidates_dict, source):
        """Kelimeyi işle ve uygunsa candidates'e ekle"""
        if not t or t == token:
            return
        
        t_lower = tr_lower(t)
        
        # Normalize dene
        norm = zemb.normalize(t_lower)
        if norm and norm != t_lower:
            if arızalar_içinde(norm):
                dist = weighted_edit_distance(token, norm)
                in_dict = True
                if norm not in candidates_dict or candidates_dict[norm]['distance'] > dist:
                    candidates_dict[norm] = {
                        'distance': dist,
                        'in_dict': in_dict,
                        'source': f"{source}+normalize"
                    }
            
            lemma_norm = zemb.get_lemma(norm)
            if lemma_norm and arızalar_içinde(lemma_norm):
                dist = weighted_edit_distance(token, lemma_norm)
                in_dict = True
                if lemma_norm not in candidates_dict or candidates_dict[lemma_norm]['distance'] > dist:
                    candidates_dict[lemma_norm] = {
                        'distance': dist,
                        'in_dict': in_dict,
                        'source': f"{source}+normalize+lemma"
                    }
        
        # Direkt kontrol
        if arızalar_içinde(t_lower):
            dist = weighted_edit_distance(token, t_lower)
            in_dict = True
            if t_lower not in candidates_dict or candidates_dict[t_lower]['distance'] > dist:
                candidates_dict[t_lower] = {
                    'distance': dist,
                    'in_dict': in_dict,
                    'source': source
                }
        
        # Lemma kontrol
        lemma = zemb.get_lemma(t_lower)
        if lemma and lemma != t_lower:
            if arızalar_içinde(lemma):
                dist = weighted_edit_distance(token, lemma)
                in_dict = True
                if lemma not in candidates_dict or candidates_dict[lemma]['distance'] > dist:
                    candidates_dict[lemma] = {
                        'distance': dist,
                        'in_dict': in_dict,
                        'source': f"{source}+lemma"
                    }
        
        # Suggest dene
        suggests = zemb.suggest(t_lower)
        if suggests:
            for s in suggests[:5]:  # İlk 5 öneri
                s_lower = tr_lower(s)
                if s_lower != t_lower and arızalar_içinde(s_lower):
                    dist = weighted_edit_distance(token, s_lower)
                    in_dict = True
                    if s_lower not in candidates_dict or candidates_dict[s_lower]['distance'] > dist:
                        candidates_dict[s_lower] = {
                            'distance': dist,
                            'in_dict': in_dict,
                            'source': f"{source}+suggest"
                        }
    
    token_lower = tr_lower(token)
    candidates_dict = {}  # {word: {'distance': float, 'in_dict': bool, 'source': str}}
    
    # === 1. Önce orijinal token'ı kontrol et ===
    işle_ve_ekle(token_lower, candidates_dict, "original")
    
    # === 2. Diacritic varyantları ===
    diacritic_variants = generate_diacritic_variants(token_lower, zemb, limit=5)
    for variant in diacritic_variants:
        işle_ve_ekle(variant, candidates_dict, "diacritic")
    
    # === 3. Zemberek önerileri (direkt) ===
    zemberek_suggests = zemb.suggest(token_lower)
    if zemberek_suggests:
        for s in zemberek_suggests[:10]:  # İlk 10 öneri
            işle_ve_ekle(s, candidates_dict, "zemberek_suggest")
    
    # === 4. Normalize ===
    normalized = zemb.normalize(token_lower)
    if normalized and normalized != token_lower:
        işle_ve_ekle(normalized, candidates_dict, "normalize")
    
    # === 5. Domain sözlüğünden Levenshtein ile yakın kelimeler ===
    for w in lemma2id:
        if abs(len(w) - len(token_lower)) <= 2:
            dist = weighted_edit_distance(w, token_lower)
            if dist <= 2:
                if w not in candidates_dict or candidates_dict[w]['distance'] > dist:
                    candidates_dict[w] = {
                        'distance': dist,
                        'in_dict': True,
                        'source': 'levenshtein'
                    }
    
    # === 6. Sadece geçerli Türkçe kelimeleri filtrele ===
    valid_candidates = {}
    for word, info in candidates_dict.items():
        if zemb.analyze_valid(word):
            valid_candidates[word] = info
    
    # === 7. Sırala: Önce sözlüktekiler, sonra edit distance ===
    sorted_candidates = sorted(
        valid_candidates.items(),
        key=lambda x: (
            not x[1]['in_dict'],  # False (sözlükte) önce, True (değil) sonra
            x[1]['distance']       # Sonra mesafeye göre
        )
    )
    
    # Sadece kelimeleri döndür
    result = [word for word, info in sorted_candidates]
    
    return result[:10]  # En fazla 10 aday

def arızaları_al(filename="arızalar.txt"):
    '''
    Parameters
    ----------
    filename : TYPE, optional
        arıza dosyası. The default is "arızalar.txt".

    Returns
    -------
    word_dict : TYPE
        arızalar sözlüğü.

    '''
    # sonuç sözlüğü
    word_list = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # sadece '-' ile başlayan satırları al
            if line.startswith("-"):
                # baştaki '-' ve boşlukları temizle
                phrase = line.lstrip("-").strip()
                # kelimelere ayır
                # words = phrase.split()
                # for w in words:
                #     # sözlükte varsa artır, yoksa ekle
                #     word_dict[w] = word_dict.get(w, 0) + 1
                
                word_list.append(tr_lower(phrase))

    return word_list

def arızaları_işle(words, zemb):

    word2id = {}
    id2word = {}
    phrase2ids = {}
    ids2phrase = {}
    
    current_id = 1
    
    for phrase in words:  # words: dosyadan okunan tüm arıza cümleleri
        ids = []
        
        phrase = tr_lower(phrase)
        
        for w in phrase.split():
            # lemma bul
            analysis = zemb.analyze_sentence(tr_lower(w))
            if analysis and "lemmas" in analysis[0] and analysis[0]["lemmas"]:
                kok = analysis[0]["lemmas"][0]
            
            if kok == "UNK":
                kok = w
                
            kok = tr_lower(kok)
            # köke ID ata
            if kok not in word2id:
                word2id[kok] = current_id
                id2word[current_id] = kok
                current_id += 1
    
            ids.append(word2id[kok])
    
        # phrase -> ids
        phrase2ids[phrase] = ids
        # ids (tuple) -> phrase
        ids2phrase[tuple(ids)] = phrase
    
    results = {
    "word2id": word2id,
    "phrase2ids": phrase2ids,
    "ids2phrase": ids2phrase
    }
    
    return  results

def arızaları_işle_v2(df, zemb, dokunma):
    
    # Ana veri yapıları
    lemma2id = {}  # lemma -> ID
    id2lemma = {}  # ID -> lemma
    
    # Kategori bazlı ID setleri
    category_phrase_ids = defaultdict(list)  # kategori -> [ID setleri]
    category_keyword_ids = defaultdict(list)  # kategori -> [ID setleri]
    category_all_ids = defaultdict(set)  # kategori -> tüm ID'lerin birleşimi
    
    # Reverse mapping (hızlı arama için)
    phrase_ids_to_info = {}  # frozenset(IDs) -> {kategori, phrase}
    keyword_ids_to_info = {}  # frozenset(IDs) -> {kategori, keyword}
    
    # İstatistikler
    category_stats = {}
    
    def get_lemma(word):
        """Kelimenin kökünü bul"""
        try:
            analysis = zemb.analyze_sentence(tr_lower(word))
            if analysis and len(analysis) > 0:
                if "lemmas" in analysis[0] and analysis[0]["lemmas"]:
                    lemma = analysis[0]["lemmas"][0]
                    if lemma and lemma != "UNK":
                        return tr_lower(lemma)
                    
                    else:
                        # print(analysis[0]["token"])
                        pass
        except:
            pass
        return "UNK"
    
    def is_valid_lemma(lemma):
        """
        Lemma geçerli mi kontrol et
        - Noktalama işaretlerini filtrele
        - Çok kısa kelimeleri filtrele
        - Sayıları filtrele (opsiyonel)
        
        Args:
            lemma: Kontrol edilecek lemma
        
        Returns:
            bool: Geçerli ise True
        """
        if not lemma or lemma == "UNK":
            return False
        
        # Sadece noktalama işareti mi?
        if lemma in string.punctuation:
            return False
        
        # Türkçe noktalama işaretleri
        turkish_punctuation = '!"#$%&\'()*+,.:;<=>?@[\\]^_`{|}~'
        if lemma in turkish_punctuation:
            return False
        
        # Tamamı noktalama mı?
        if all(c in turkish_punctuation for c in lemma):
            return False
        
        # Çok kısa (opsiyonel - 1 karakterlik kelimeleri filtrele)
        # if len(lemma) < 2:
        #     return False
        
        # Sadece rakam mı? (opsiyonel)
        # if lemma.isdigit():
        #     return False
        
        return True
    
    def _add_lemma_to_lists(lemma, original_word, lemmas, ids):
        """
        Lemma'yı listelere ekle (helper method)
        """
        if lemma == "UNK":
            # Fuzzy match dene
            best_match = _fuzzy_match(original_word)
            if best_match:
                lemma = best_match
            else:
                lemma = original_word
        
        lemmas.append(lemma)
        
        # ID ekle (eğer sözlükte varsa)
        if lemma in lemma2id:
            ids.append(lemma2id[lemma])
    
    def fuzzy_match(word, threshold=3):
        """
        Kelime için en yakın eşleşmeyi bul (fuzzy matching)
        
        Args:
            word: Aranacak kelime
            dokunma: Aranacak kelime listesi
            threshold: Maksimum düzenleme mesafesi
        
        Returns:
            En yakın kelime veya None
        """
        # Array/Series/List kontrolü
        if dokunma is None or len(dokunma) == 0:
            return None
        
        best_match = None
        min_dist = float("inf")
        
        for candidate in dokunma:
            dist = weighted_edit_distance(candidate, word)
            
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
        
        # Eşik kontrolü
        if min_dist <= threshold:
            return best_match
        
        return None

    
    def build_id_structures():
        """
        Ana veri yapılarını oluştur
        - Tire/slash desteği
        - Fuzzy matching
        - Noktalama filtreleme ✨ YENİ
        """
        print("🔧 ID yapıları oluşturuluyor...")
        
        current_id = 1
        
        # === 1. ADIM: Tüm unique lemmaları topla ve ID ata ===
        all_texts = set()
        for _, row in df.iterrows():
            key_phrase = tr_lower(row['Key Phrase'])
            keywords = tr_lower(row['Keywords'])
            
            if pd.notna(key_phrase):
                all_texts.add(key_phrase)
            if pd.notna(keywords):
                all_texts.add(keywords)
        
        # Lemma -> ID mapping
        for text in all_texts:
            words = text.split()
            
            for word in words:
                if not word:
                    continue
                
                # Tire veya slash içeriyor mu?
                if "-" in word or "/" in word:
                    separator = "-" if "-" in word else "/"
                    parts = word.split(separator)
                    
                    for part in parts:
                        part = part.strip()
                        if part:
                            lemma = get_lemma(part)
                            
                            # UNK ise fuzzy match
                            if lemma == "UNK":
                                best = fuzzy_match(part)
                                lemma = best if best else part
                            
                            # ✨ YENİ: Geçerli lemma mı kontrol et
                            if is_valid_lemma(lemma) and lemma not in lemma2id:
                                lemma2id[lemma] = current_id
                                id2lemma[current_id] = lemma
                                current_id += 1
                else:
                    # Normal kelime
                    lemma = get_lemma(word)
                    
                    # UNK ise fuzzy match
                    if lemma == "UNK":
                        best = fuzzy_match(word)
                        lemma = best if best else word
                    
                    # ✨ YENİ: Geçerli lemma mı kontrol et
                    if is_valid_lemma(lemma) and lemma not in lemma2id:
                        lemma2id[lemma] = current_id
                        id2lemma[current_id] = lemma
                        current_id += 1
        
        print(f"   ✅ {len(lemma2id)} benzersiz lemma bulundu")
        
        # === 2. ADIM: Kategorilere göre UNIQUE ID setleri oluştur ===
        seen_phrases = set()
        seen_keywords = set()
        
        for _, row in df.iterrows():
            kategori = row['Arıza Kategorisi']
            key_phrase = tr_lower(row['Key Phrase'])
            keywords = tr_lower(row['Keywords'])
            
            if pd.isna(kategori):
                continue
            
            # === KEY PHRASE İŞLEME ===
            if pd.notna(key_phrase):
                phrase_key = (kategori, key_phrase)
                
                if phrase_key not in seen_phrases:
                    seen_phrases.add(phrase_key)
                    
                    # Text'i ID'lere çevir (tire/slash ve fuzzy match ile)
                    phrase_ids = text_to_id_list(key_phrase)
                    
                    if phrase_ids:
                        phrase_id_set = frozenset(phrase_ids)
                        
                        category_phrase_ids[kategori].append({
                            'ids': phrase_id_set,
                            'ids_list': phrase_ids,
                            'phrase': key_phrase,
                            'length': len(phrase_ids)
                        })
                        
                        if phrase_id_set not in phrase_ids_to_info:
                            phrase_ids_to_info[phrase_id_set] = {
                                'kategori': kategori,
                                'phrase': key_phrase,
                                'type': 'phrase'
                            }
                        
                        category_all_ids[kategori].update(phrase_ids)
            
            # === KEYWORD İŞLEME ===
            if pd.notna(keywords):
                keyword_key = (kategori, keywords)
                
                if keyword_key not in seen_keywords:
                    seen_keywords.add(keyword_key)
                    
                    # Text'i ID'lere çevir
                    keyword_ids = text_to_id_list(keywords)
                    
                    if keyword_ids:
                        keyword_id_set = frozenset(keyword_ids)
                        
                        category_keyword_ids[kategori].append({
                            'ids': keyword_id_set,
                            'ids_list': keyword_ids,
                            'keyword': keywords,
                            'length': len(keyword_ids)
                        })
                        
                        if keyword_id_set not in keyword_ids_to_info:
                            keyword_ids_to_info[keyword_id_set] = {
                                'kategori': kategori,
                                'keyword': keywords,
                                'type': 'keyword'
                            }
                        
                        category_all_ids[kategori].update(keyword_ids)
        
        # === 3. ADIM: İstatistikler ===
        for kategori in category_phrase_ids.keys():
            category_stats[kategori] = {
                'total_phrases': len(category_phrase_ids[kategori]),
                'total_keywords': len(category_keyword_ids[kategori]),
                'unique_ids': len(category_all_ids[kategori])
            }
        
    def _split_text_to_words(text):
        """
        Metni kelimelere ayır (tire/slash desteği ile)
        
        Args:
            text: İşlenecek metin
        
        Returns:
            List of words
        """
        if not text:
            return []
        
        words = []
        
        for token in text.split():
            # Tire veya slash varsa ayır
            if "-" in token:
                parts = token.split("-")
                words.extend([p.strip() for p in parts if p.strip()])
            elif "/" in token:
                parts = token.split("/")
                words.extend([p.strip() for p in parts if p.strip()])
            else:
                words.append(token)
        
        return words
    
    
    def text_to_id_list(text):
        """
        Text'i ID listesine çevir
        - Noktalama filtreleme ✨ YENİ
        """
        if not text:
            return []
        
        ids = []
        words = text.split()
        
        for word in words:
            if not word:
                continue
            
            # Tire veya slash içeriyor mu?
            if "-" in word or "/" in word:
                separator = "-" if "-" in word else "/"
                parts = word.split(separator)
                
                for part in parts:
                    part = part.strip()
                    if part:
                        lemma = get_lemma(part)
                        
                        if lemma == "UNK":
                            best = fuzzy_match(part)
                            lemma = best if best else part
                        
                        # ✨ YENİ: Geçerli lemma kontrolü
                        if is_valid_lemma(lemma) and lemma in lemma2id:
                            ids.append(lemma2id[lemma])
            else:
                lemma = get_lemma(word)
                
                if lemma == "UNK":
                    best = fuzzy_match(word)
                    lemma = best if best else word
                
                # ✨ YENİ: Geçerli lemma kontrolü
                if is_valid_lemma(lemma) and lemma in lemma2id:
                    ids.append(lemma2id[lemma])
        
        return ids
    
    
    def text_to_lemma_ids(text):
        """
        Metni lemma ID'lerine çevir
        - Noktalama filtreleme ✨ YENİ
        """
        text = tr_lower(text)
        words = text.split()
        ids = []
        lemmas = []
        
        for word in words:
            if not word:
                continue
            
            # Tire/slash içeriyor mu?
            if "-" in word or "/" in word:
                separator = "-" if "-" in word else "/"
                parts = word.split(separator)
                
                for part in parts:
                    part = part.strip()
                    if part:
                        lemma = get_lemma(part)
                        
                        if lemma == "UNK":
                            best = fuzzy_match(part)
                            lemma = best if best else part
                        
                        # ✨ YENİ: Geçerli lemma kontrolü
                        if is_valid_lemma(lemma):
                            lemmas.append(lemma)
                            if lemma in lemma2id:
                                ids.append(lemma2id[lemma])
            else:
                lemma = get_lemma(word)
                
                if lemma == "UNK":
                    best = fuzzy_match(word)
                    lemma = best if best else word
                
                # ✨ YENİ: Geçerli lemma kontrolü
                if is_valid_lemma(lemma):
                    lemmas.append(lemma)
                    if lemma in lemma2id:
                        ids.append(lemma2id[lemma])
        
        return ids, lemmas, words
        
    
    build_id_structures()
    
    return {
            'lemma2id': lemma2id,
            'id2lemma': id2lemma,
            'category_phrase_ids': category_phrase_ids,
            'category_keyword_ids': category_keyword_ids,
            'category_all_ids': category_all_ids,
            'phrase_ids_to_info': phrase_ids_to_info,
            'keyword_ids_to_info': keyword_ids_to_info,
            'category_stats': category_stats
        }


@dataclass
class ArızaStructures:
    """
    Güncellenmiş arıza veri yapıları
    - Şebeke unsuru bazlı kategori organizasyonu
    """
    # Lemma mappings
    lemma2id: Dict[str, int] = field(default_factory=dict)
    id2lemma: Dict[int, str] = field(default_factory=dict)
    
    # ✨ YENİ: Cause Code → Şebeke Unsuru mapping
    cause_code_to_unsur: Dict[str, str] = field(default_factory=dict)
    
    # ✨ YENİ: Şebeke Unsuru → Kategoriler
    unsur_to_categories: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Kategori bazlı ID setleri (ESKİ yapı korunuyor)
    category_phrase_ids: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    category_keyword_ids: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    category_all_ids: Dict[str, Set] = field(default_factory=lambda: defaultdict(set))
    
    # ✨ YENİ: Şebeke Unsuru bazlı erişim (hızlı filtreleme için)
    unsur_category_phrase_ids: Dict[str, Dict[str, List]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    unsur_category_keyword_ids: Dict[str, Dict[str, List]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    
    # Reverse mapping
    phrase_ids_to_info: Dict[frozenset, Dict] = field(default_factory=dict)
    keyword_ids_to_info: Dict[frozenset, Dict] = field(default_factory=dict)
    
    # İstatistikler
    category_stats: Dict[str, Dict] = field(default_factory=dict)
    unsur_stats: Dict[str, Dict] = field(default_factory=dict)  # ✨ YENİ


def load_cause_code_mapping(cause_code_path: str) -> Dict[str, str]:
    """
    Cause Code → Şebeke Unsuru mapping'i yükle
    - "-" değerlerini de işle
    
    Args:
        cause_code_path: Excel dosya yolu
    
    Returns:
        Dict: {cause_code: şebeke_unsuru}
    """
    df = pd.read_excel(cause_code_path)
    
    # Column names'i normalize et
    df.columns = df.columns.str.strip()
    
    mapping = {}
    for _, row in df.iterrows():
        cause_code = str(row['cause code']).strip()
        sebeke_unsuru = str(row['Şebeke Unsuru']).strip()
        
        if pd.notna(cause_code):
            # ✨ "-" veya boş değerleri de mapping'e ekle
            if pd.isna(sebeke_unsuru) or sebeke_unsuru == "" or sebeke_unsuru == "-":
                mapping[cause_code] = "-"  # Özel işaret
            else:
                mapping[cause_code] = sebeke_unsuru
    
    print(f"✅ {len(mapping)} cause code mapping yüklendi")
    
    # İstatistik
    global_count = sum(1 for v in mapping.values() if v == "-")
    filtered_count = len(mapping) - global_count
    
    print(f"   - {filtered_count} filtreli (spesifik şebeke unsuru)")
    print(f"   - {global_count} global (tüm kategorilerde arama)")
    
    return mapping


def arızaları_işle_v3(
    arizalar_df: pd.DataFrame,
    cause_code_df_path: str,
    zemb,
    dokunma: List[str]
) -> ArızaStructures:
    """
    Arıza verilerini işle - V3 (Şebeke Unsuru bazlı)
    
    Args:
        arizalar_df: Arıza verileri (Key Phrase, Keywords, Şebeke Unsuru, Arıza Kategorisi)
        cause_code_df_path: Cause code mapping Excel path
        zemb: Zemberek analyzer
        dokunma: Fuzzy match için kelime listesi
    
    Returns:
        ArızaStructures
    """
    structures = ArızaStructures()
    
    # 1. Cause Code mapping'i yükle
    structures.cause_code_to_unsur = load_cause_code_mapping(cause_code_df_path)
    
    print("🔧 ID yapıları oluşturuluyor (Şebeke Unsuru bazlı)...")
    
    current_id = 1
    
    # Helper functions (önceki gibi)
    def get_lemma(word):
        """Kelimenin kökünü bul"""
        try:
            analysis = zemb.analyze_sentence(tr_lower(word))
            if analysis and len(analysis) > 0:
                if "lemmas" in analysis[0] and analysis[0]["lemmas"]:
                    lemma = analysis[0]["lemmas"][0]
                    if lemma and lemma != "UNK":
                        return tr_lower(lemma)
        except:
            pass
        return "UNK"
    
    def is_valid_lemma(lemma):
        """Lemma geçerli mi kontrol et"""
        if not lemma or lemma == "UNK":
            return False
        
        turkish_punctuation = '!"#$%&\'()*+,.:;<=>?@[\\]^_`{|}~'
        if lemma in turkish_punctuation or all(c in turkish_punctuation for c in lemma):
            return False
        
        return True
    
    def fuzzy_match(word, threshold=3):
        """Fuzzy matching"""
        if dokunma is None or len(dokunma) == 0:
            return None
        
        best_match = None
        min_dist = float("inf")
        
        for candidate in dokunma:
            dist = weighted_edit_distance(candidate, word)
            if dist < min_dist:
                min_dist = dist
                best_match = candidate
        
        if min_dist <= threshold:
            return best_match
        
        return None
    
    # === 1. ADIM: Lemma → ID mapping oluştur ===
    all_texts = set()
    for _, row in arizalar_df.iterrows():
        key_phrase = tr_lower(row['Key Phrase'])
        keywords = tr_lower(row['Keywords'])
        
        if pd.notna(key_phrase):
            all_texts.add(key_phrase)
        if pd.notna(keywords):
            all_texts.add(keywords)
    
    for text in all_texts:
        for word in text.split():
            if not word or '-' in word or '/' in word:
                # Tire/slash varsa parçala
                if '-' in word:
                    parts = word.split('-')
                elif '/' in word:
                    parts = word.split('/')
                else:
                    continue
                
                for part in parts:
                    part = part.strip()
                    if part:
                        lemma = get_lemma(part)
                        if lemma == "UNK":
                            lemma = fuzzy_match(part) or part
                        
                        if is_valid_lemma(lemma) and lemma not in structures.lemma2id:
                            structures.lemma2id[lemma] = current_id
                            structures.id2lemma[current_id] = lemma
                            current_id += 1
            else:
                lemma = get_lemma(word)
                if lemma == "UNK":
                    lemma = fuzzy_match(word) or word
                
                if is_valid_lemma(lemma) and lemma not in structures.lemma2id:
                    structures.lemma2id[lemma] = current_id
                    structures.id2lemma[current_id] = lemma
                    current_id += 1
    
    print(f"   ✅ {len(structures.lemma2id)} benzersiz lemma bulundu")
    
    # === 2. ADIM: Text → ID list helper ===
    def text_to_id_list(text):
        """Text'i ID listesine çevir"""
        if not text:
            return []
        
        ids = []
        for word in text.split():
            if not word:
                continue
            
            # Tire/slash varsa parçala
            if "-" in word or "/" in word:
                separator = "-" if "-" in word else "/"
                parts = word.split(separator)
                
                for part in parts:
                    part = part.strip()
                    if part:
                        lemma = get_lemma(part)
                        if lemma == "UNK":
                            lemma = fuzzy_match(part) or part
                        
                        if is_valid_lemma(lemma) and lemma in structures.lemma2id:
                            ids.append(structures.lemma2id[lemma])
            else:
                lemma = get_lemma(word)
                if lemma == "UNK":
                    lemma = fuzzy_match(word) or word
                
                if is_valid_lemma(lemma) and lemma in structures.lemma2id:
                    ids.append(structures.lemma2id[lemma])
        
        return ids
    
    # === 3. ADIM: Şebeke Unsuru + Kategori bazlı yapı oluştur ===
    seen_phrases = set()
    seen_keywords = set()
    
    for _, row in arizalar_df.iterrows():
        sebeke_unsuru = row['Şebeke Unsuru']
        kategori = row['Arıza Kategorisi']
        key_phrase = tr_lower(row['Key Phrase'])
        keywords = tr_lower(row['Keywords'])
        
        if pd.isna(sebeke_unsuru) or pd.isna(kategori):
            continue
        
        # ✨ Şebeke Unsuru → Kategori mapping
        structures.unsur_to_categories[sebeke_unsuru].add(kategori)
        
        # === KEY PHRASE İŞLEME ===
        if pd.notna(key_phrase):
            phrase_key = (sebeke_unsuru, kategori, key_phrase)
            
            if phrase_key not in seen_phrases:
                seen_phrases.add(phrase_key)
                
                phrase_ids = text_to_id_list(key_phrase)
                
                if phrase_ids:
                    phrase_id_set = frozenset(phrase_ids)
                    
                    phrase_info = {
                        'ids': phrase_id_set,
                        'ids_list': phrase_ids,
                        'phrase': key_phrase,
                        'length': len(phrase_ids)
                    }
                    
                    # Global yapı (eski)
                    structures.category_phrase_ids[kategori].append(phrase_info)
                    
                    # ✨ YENİ: Şebeke Unsuru bazlı yapı
                    structures.unsur_category_phrase_ids[sebeke_unsuru][kategori].append(phrase_info)
                    
                    # Reverse mapping
                    if phrase_id_set not in structures.phrase_ids_to_info:
                        structures.phrase_ids_to_info[phrase_id_set] = {
                            'kategori': kategori,
                            'sebeke_unsuru': sebeke_unsuru,
                            'phrase': key_phrase,
                            'type': 'phrase'
                        }
                    
                    structures.category_all_ids[kategori].update(phrase_ids)
        
        # === KEYWORD İŞLEME ===
        if pd.notna(keywords):
            keyword_key = (sebeke_unsuru, kategori, keywords)
            
            if keyword_key not in seen_keywords:
                seen_keywords.add(keyword_key)
                
                keyword_ids = text_to_id_list(keywords)
                
                if keyword_ids:
                    keyword_id_set = frozenset(keyword_ids)
                    
                    keyword_info = {
                        'ids': keyword_id_set,
                        'ids_list': keyword_ids,
                        'keyword': keywords,
                        'length': len(keyword_ids)
                    }
                    
                    # Global yapı (eski)
                    structures.category_keyword_ids[kategori].append(keyword_info)
                    
                    # ✨ YENİ: Şebeke Unsuru bazlı yapı
                    structures.unsur_category_keyword_ids[sebeke_unsuru][kategori].append(keyword_info)
                    
                    # Reverse mapping
                    if keyword_id_set not in structures.keyword_ids_to_info:
                        structures.keyword_ids_to_info[keyword_id_set] = {
                            'kategori': kategori,
                            'sebeke_unsuru': sebeke_unsuru,
                            'keyword': keywords,
                            'type': 'keyword'
                        }
                    
                    structures.category_all_ids[kategori].update(keyword_ids)
    
    # === 4. ADIM: İstatistikler ===
    for kategori in structures.category_phrase_ids.keys():
        structures.category_stats[kategori] = {
            'total_phrases': len(structures.category_phrase_ids[kategori]),
            'total_keywords': len(structures.category_keyword_ids[kategori]),
            'unique_ids': len(structures.category_all_ids[kategori])
        }
    
    # ✨ Şebeke Unsuru istatistikleri
    for unsur in structures.unsur_to_categories.keys():
        total_categories = len(structures.unsur_to_categories[unsur])
        total_phrases = sum(
            len(structures.unsur_category_phrase_ids[unsur][cat])
            for cat in structures.unsur_to_categories[unsur]
        )
        total_keywords = sum(
            len(structures.unsur_category_keyword_ids[unsur][cat])
            for cat in structures.unsur_to_categories[unsur]
        )
        
        structures.unsur_stats[unsur] = {
            'total_categories': total_categories,
            'total_phrases': total_phrases,
            'total_keywords': total_keywords,
            'categories': list(structures.unsur_to_categories[unsur])
        }
    
    print(f"   ✅ {len(structures.unsur_to_categories)} şebeke unsuru")
    print(f"   ✅ {len(structures.category_phrase_ids)} kategori işlendi")
    print(f"   ✅ {len(seen_phrases)} unique phrase")
    print(f"   ✅ {len(seen_keywords)} unique keyword")
    
    return structures    



def kelimeleri_al(filename):
    '''
    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    kelimeler = set()
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            kelimeler.add(line)
            
    return kelimeler

    
def find_file(filename, start_dir=None):
    """
    Belirtilen dosya adını (filename) start_dir altında arar.
    start_dir verilmezse, script'in bulunduğu klasörü kullanır.
    Bulursa dosyanın tam yolunu döner, bulamazsa None döner.
    """
    if start_dir is None:
        
        start_dir = os.path.dirname(os.path.abspath(__file__))

    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None
        
def softmax_normalize(beam_nodes):
    """
    Skorları softmax ile normalize et (toplamları 1 olur)
    """
    scores = np.array([node.score for node in beam_nodes])
    
    # Softmax: exp(x) / sum(exp(x))
    exp_scores = np.exp(scores)
    normalized_scores = exp_scores / np.sum(exp_scores)
    
    # Güncellenmiş node'ları döndür
    for node, norm_score in zip(beam_nodes, normalized_scores):
        node.normalized_score = norm_score
    
    return beam_nodes