# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:00:41 2025

@author: vural.bayrakli
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Set, Optional
import pandas as pd
from tqdm import tqdm


@dataclass
class MorphosemanticToken:
    """Morfosemantik bilgiyle zenginleştirilmiş token"""
    lemma: str              # "kapamak"
    pos: str                # "Verb", "Noun", "Adj" vb.
    is_negative: bool       # True/False (olumsuz mu?)
    morphemes: List[str]    # ["Pass", "Neg", "Past", "A3sg"]
    surface_form: str       # "kapanmadı"
    
    def to_key(self) -> str:
        """Unique key oluştur"""
        neg_marker = "_NEG" if self.is_negative else ""
        return f"{self.lemma}_{self.pos}{neg_marker}"
    
    def to_display_lemma(self) -> str:
        """
        Görüntüleme için lemma (negatif eki dahil)
        "kapamak" → "kapamamak" (olumsuz ise)
        """
        if not self.is_negative or self.pos != "Verb":
            return self.lemma
        
        # Türkçe olumsuz eki: -ma/-me (son ünlüye göre)
        if self.lemma.endswith("mak"):
            stem = self.lemma[:-3]  # "kapa"
            # Son ünlüyü bul
            last_vowel = None
            for char in reversed(stem):
                if char in "aeıioöuü":
                    last_vowel = char
                    break
            
            if last_vowel in "aıou":
                return stem + "mamak"
            else:
                return stem + "memek"
        
        elif self.lemma.endswith("mek"):
            stem = self.lemma[:-3]
            last_vowel = None
            for char in reversed(stem):
                if char in "aeıioöuü":
                    last_vowel = char
                    break
            
            if last_vowel in "aıou":
                return stem + "mamak"
            else:
                return stem + "memek"
        
        # Fallback
        return self.lemma + "_NEG"
    
    def __hash__(self):
        return hash(self.to_key())
    
    def __eq__(self, other):
        return self.to_key() == other.to_key()


@dataclass
class ArızaStructures:
    """
    Morfosemantik bilgiyle zenginleştirilmiş arıza yapıları
    """
    # ✨ Morfosemantik ID mappings
    token_to_id: Dict[str, int] = field(default_factory=dict)  # "kapamak_Verb_NEG" → ID
    id_to_token: Dict[int, MorphosemanticToken] = field(default_factory=dict)
    
    # Geriye uyumluluk için
    lemma2id: Dict[str, int] = field(default_factory=dict)
    id2lemma: Dict[int, str] = field(default_factory=dict)
    
    # Cause Code → Şebeke Unsuru mapping
    cause_code_to_unsur: Dict[str, str] = field(default_factory=dict)
    
    # Şebeke Unsuru → Kategoriler
    unsur_to_categories: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Kategori bazlı ID setleri
    category_phrase_ids: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    category_keyword_ids: Dict[str, List] = field(default_factory=lambda: defaultdict(list))
    category_all_ids: Dict[str, Set] = field(default_factory=lambda: defaultdict(set))
    
    # Şebeke Unsuru bazlı erişim
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
    unsur_stats: Dict[str, Dict] = field(default_factory=dict)


def detect_true_negation(morphemes: List[str], pos: str) -> bool:
    """
    Gerçek olumsuzluğu tespit et - Inf2 ve Imp+Neg'i hariç tut
    
    Rules:
    1. Inf2 varsa → NEG değil (mastar: "oturma", "çalışma")
    2. Neg + Imp → NEG değil (emir: "oturma!", "çalışma!")
    3. Neg + Past/Pres/Fut → Gerçek olumsuz ("oturmadı", "oturmuyor")
    4. Sadece Neg → Belirsiz, False döndür (güvenli taraf)
    
    Args:
        morphemes: Morpheme listesi
        pos: Part of speech
    
    Returns:
        True sadece gerçek olumsuzsa (geçmiş/şimdiki/gelecek zaman)
    """
    if "Neg" not in morphemes:
        return False
    
    # Rule 1: Inf2 (mastar -ma/-me) varsa NEG değil
    if "Inf2" in morphemes:
        return False
    
    # Rule 2: Inf1 varsa da NEG değil (başka mastar formu: -mak/-mek)
    if "Inf1" in morphemes:
        return False
    
    # Rule 3: Neg + Imp birlikte varsa NEG değil (olumsuz emir ayır)
    if "Neg" in morphemes and "Imp" in morphemes:
        return False
    
    # Rule 4: Gerçek olumsuz yapılar (zaman/kip ekleri)
    real_negative_markers = {
        "Past",      # -madı/-medi
        "Pres",      # -mıyor/-miyor/-muyor/-müyor
        "Fut",       # -mayacak/-meyecek
        "Prog1",     # -makta/-mekte
        "Prog2",     # -mıyor/-miyor
        "Aor",       # -maz/-mez
        "Opt",       # -maya/-meye
        "Cond",      # -masa/-mese
        "Neces",     # -malı/-meli
    }
    
    # Zaman/kip ekleri varsa gerçek olumsuz
    if any(marker in morphemes for marker in real_negative_markers):
        return True
    
    # Hiçbiri yoksa güvenli taraf: NEG değil
    return False


def process_text_with_context_smart(text: str, zemb) -> List[MorphosemanticToken]:
    """
    Text'i akıllıca analiz et - ambiguity'yi çöz
    
    Strateji:
    1. Tam metni analiz et (bağlam korunur)
    2. Inf2 (mastar -ma/-me) varsa → NEG değil!
    3. Neg + Imp birlikte varsa → NEG değil (olumsuz emir)
    4. Neg + Zaman eki varsa → Gerçek olumsuz
    """
    # Tire ve slash'leri boşluğa çevir
    normalized_text = text.replace('-', ' ').replace('/', ' ')
    
    # Tüm metni analiz et
    try:
        resp = zemb.analyze_sentence_full(normalized_text)
        
        tokens = []
        
        for result in resp.results:
            if not result.best:
                tokens.append(MorphosemanticToken(
                    lemma=tr_lower(result.token),
                    pos="Unknown",
                    is_negative=False,
                    morphemes=[],
                    surface_form=result.token
                ))
                continue
            
            best = result.best
            lemma = best.dictionaryItem.lemma if best.dictionaryItem.lemma else result.token
            pos = best.pos if best.pos else "Unknown"
            morphemes = [m.morpheme for m in best.morphemes if m.morpheme]
            
            # ✅ AKILLI OLUMSUZLUK TESPİTİ
            is_negative = detect_true_negation(morphemes, pos)
            
            tokens.append(MorphosemanticToken(
                lemma=tr_lower(lemma),
                pos=pos,
                is_negative=is_negative,
                morphemes=morphemes,
                surface_form=result.token
            ))
        
        return tokens
        
    except Exception as e:
        print(f"⚠️ Analiz hatası: {text} - {e}")
        return [
            MorphosemanticToken(
                lemma=tr_lower(word),
                pos="Unknown",
                is_negative=False,
                morphemes=[],
                surface_form=word
            )
            for word in text.split()
        ]


def get_morphosemantic_id(token: MorphosemanticToken, structures: ArızaStructures, 
                          auto_create: bool = True) -> Optional[int]:
    """
    Morfosemantik token için ID al veya oluştur
    
    Args:
        token: MorphosemanticToken
        structures: ArızaStructures
        auto_create: Yoksa yeni ID oluştur mu?
    
    Returns:
        ID veya None
    """
    key = token.to_key()
    
    if key in structures.token_to_id:
        return structures.token_to_id[key]
    
    if auto_create:
        new_id = len(structures.token_to_id) + 1
        structures.token_to_id[key] = new_id
        structures.id_to_token[new_id] = token
        
        # Görüntüleme lemma'sını lemma2id'ye ekle
        display_lemma = token.to_display_lemma()
        if display_lemma not in structures.lemma2id:
            structures.lemma2id[display_lemma] = new_id
            structures.id2lemma[new_id] = display_lemma
        
        return new_id
    
    return None


def is_valid_token(token: MorphosemanticToken) -> bool:
    """Token geçerli mi kontrol et"""
    if not token.lemma or token.lemma == "UNK":
        return False
    
    turkish_punctuation = '!"#$%&\'()*+,.:;<=>?@[\\]^_`{|}~'
    if token.lemma in turkish_punctuation or all(c in turkish_punctuation for c in token.lemma):
        return False
    
    return True


def fuzzy_match(word: str, candidates: List[str], threshold: int = 3) -> Optional[str]:
    """Fuzzy matching"""
    if len(candidates) == 0:
        return None
    
    from Levenshtein import distance
    
    best_match = None
    min_dist = float("inf")
    
    for candidate in candidates:
        dist = distance(candidate, word)
        if dist < min_dist:
            min_dist = dist
            best_match = candidate
    
    return best_match if min_dist <= threshold else None


def load_cause_code_mapping(cause_code_path: str) -> Dict[str, str]:
    """
    Cause Code → Şebeke Unsuru mapping'i yükle
    """
    df = pd.read_excel(cause_code_path)
    df.columns = df.columns.str.strip()
    
    mapping = {}
    for _, row in df.iterrows():
        cause_code = str(row['cause code']).strip()
        sebeke_unsuru = str(row['Şebeke Unsuru']).strip()
        
        if pd.notna(cause_code):
            if pd.isna(sebeke_unsuru) or sebeke_unsuru == "" or sebeke_unsuru == "-":
                mapping[cause_code] = "-"
            else:
                mapping[cause_code] = sebeke_unsuru
    
    print(f"✅ {len(mapping)} cause code mapping yüklendi")
    
    global_count = sum(1 for v in mapping.values() if v == "-")
    filtered_count = len(mapping) - global_count
    
    print(f"   - {filtered_count} filtreli (spesifik şebeke unsuru)")
    print(f"   - {global_count} global (tüm kategorilerde arama)")
    
    return mapping


def tr_lower(text):
    """Türkçe karakterlere duyarlı küçük harf"""
    if pd.isna(text):
        return ""
    return str(text).replace('I', 'ı').replace('İ', 'i').lower()


def arızaları_işle_v4_morphosemantic(
    arizalar_df: pd.DataFrame,
    cause_code_df_path: str,
    zemb,
    dokunma: List[str] = None
) -> ArızaStructures:
    """
    Arıza verilerini morfosemantik analizle işle - V4 (Ambiguity fix)
    
    ✅ Her zaman TAM phrase/keyword'ü birlikte analiz et
    ✅ Inf2 (mastar) ve Imp+Neg (olumsuz emir) NEG olarak işaretlenmiyor
    ✅ Sadece zaman ekli olumsuz fiiller (Past/Pres/Fut) NEG
    
    Args:
        arizalar_df: Arıza verileri (Key Phrase, Keywords, Şebeke Unsuru, Arıza Kategorisi)
        cause_code_df_path: Cause code mapping Excel path
        zemb: Zemberek analyzer
        dokunma: Fuzzy match için kelime listesi (opsiyonel)
    
    Returns:
        ArızaStructures
    """
    structures = ArızaStructures()
    
    # 1. Cause Code mapping'i yükle
    structures.cause_code_to_unsur = load_cause_code_mapping(cause_code_df_path)
    
    print("\n🔧 Morfosemantik ID yapıları oluşturuluyor (akıllı olumsuzluk tespiti)...")
    
    # === ADIM 1: Tüm metinleri topla ===
    all_texts = set()
    for _, row in arizalar_df.iterrows():
        key_phrase = tr_lower(row['Key Phrase'])
        keywords = tr_lower(row['Keywords'])
        
        if pd.notna(key_phrase):
            all_texts.add(key_phrase)
        if pd.notna(keywords):
            all_texts.add(keywords)
    
    print(f"   📝 {len(all_texts)} unique phrase/keyword analiz ediliyor...")
    
    # === ADIM 2: Her phrase'i TAM olarak analiz et (bağlam korunur) ===
    text_to_tokens = {}
    
    for text in tqdm(all_texts, desc="Morfosemantik analiz"):
        tokens = process_text_with_context_smart(text, zemb)
        
        # Fuzzy matching (gerekirse)
        if len(dokunma):
            corrected_tokens = []
            for token in tokens:
                if token.lemma == "UNK" or token.pos == "Unknown":
                    best_match = fuzzy_match(token.surface_form, dokunma, threshold=3)
                    if best_match:
                        # Düzeltilmiş kelimeyi tekrar analiz et
                        corrected = process_text_with_context_smart(best_match, zemb)
                        if corrected:
                            corrected_tokens.append(corrected[0])
                        else:
                            corrected_tokens.append(token)
                    else:
                        corrected_tokens.append(token)
                else:
                    corrected_tokens.append(token)
            
            tokens = corrected_tokens
        
        text_to_tokens[text] = tokens
        
        # ID'leri oluştur
        for token in tokens:
            if is_valid_token(token):
                get_morphosemantic_id(token, structures, auto_create=True)
    
    print(f"   ✅ {len(structures.token_to_id)} benzersiz morfosemantik token bulundu")
    print(f"   ✅ {len(structures.lemma2id)} benzersiz display lemma bulundu")
    
    # İstatistikler
    neg_count = sum(1 for t in structures.id_to_token.values() if t.is_negative)
    inf2_count = sum(1 for t in structures.id_to_token.values() if "Inf2" in t.morphemes)
    imp_neg_count = sum(1 for t in structures.id_to_token.values() 
                        if "Neg" in t.morphemes and "Imp" in t.morphemes and not t.is_negative)
    
    print(f"   📊 Gerçek olumsuz (zaman ekli): {neg_count}")
    print(f"   📊 Mastar (Inf2): {inf2_count}")
    print(f"   📊 Olumsuz emir (Imp+Neg, NEG değil): {imp_neg_count}")
    
    # === ADIM 3: Text → ID list helper (cache kullan) ===
    def text_to_morphosemantic_id_list(text: str) -> List[int]:
        """Text'i morfosemantik ID listesine çevir (cache'den)"""
        if not text:
            return []
        
        # Cache'de var mı?
        if text in text_to_tokens:
            tokens = text_to_tokens[text]
        else:
            # Yoksa analiz et
            tokens = process_text_with_context_smart(text, zemb)
            text_to_tokens[text] = tokens
        
        # ID'leri al
        ids = []
        for token in tokens:
            if is_valid_token(token):
                token_id = get_morphosemantic_id(token, structures, auto_create=False)
                if token_id:
                    ids.append(token_id)
        
        return ids
    
    # === ADIM 4: Şebeke Unsuru + Kategori bazlı yapı oluştur ===
    seen_phrases = set()
    seen_keywords = set()
    
    print("   🏗️ Kategori yapıları oluşturuluyor...")
    
    for _, row in tqdm(arizalar_df.iterrows(), total=len(arizalar_df), desc="Kategoriler"):
        sebeke_unsuru = row['Şebeke Unsuru']
        kategori = row['Arıza Kategorisi']
        key_phrase = tr_lower(row['Key Phrase'])
        keywords = tr_lower(row['Keywords'])
        
        if pd.isna(sebeke_unsuru) or pd.isna(kategori):
            continue
        
        # Şebeke Unsuru → Kategori mapping
        structures.unsur_to_categories[sebeke_unsuru].add(kategori)
        
        # === KEY PHRASE İŞLEME ===
        if pd.notna(key_phrase):
            phrase_key = (sebeke_unsuru, kategori, key_phrase)
            
            if phrase_key not in seen_phrases:
                seen_phrases.add(phrase_key)
                
                phrase_ids = text_to_morphosemantic_id_list(key_phrase)
                
                if phrase_ids:
                    phrase_id_set = frozenset(phrase_ids)
                    
                    phrase_info = {
                        'ids': phrase_id_set,
                        'ids_list': phrase_ids,
                        'phrase': key_phrase,
                        'length': len(phrase_ids)
                    }
                    
                    structures.category_phrase_ids[kategori].append(phrase_info)
                    structures.unsur_category_phrase_ids[sebeke_unsuru][kategori].append(phrase_info)
                    
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
                
                keyword_ids = text_to_morphosemantic_id_list(keywords)
                
                if keyword_ids:
                    keyword_id_set = frozenset(keyword_ids)
                    
                    keyword_info = {
                        'ids': keyword_id_set,
                        'ids_list': keyword_ids,
                        'keyword': keywords,
                        'length': len(keyword_ids)
                    }
                    
                    structures.category_keyword_ids[kategori].append(keyword_info)
                    structures.unsur_category_keyword_ids[sebeke_unsuru][kategori].append(keyword_info)
                    
                    if keyword_id_set not in structures.keyword_ids_to_info:
                        structures.keyword_ids_to_info[keyword_id_set] = {
                            'kategori': kategori,
                            'sebeke_unsuru': sebeke_unsuru,
                            'keyword': keywords,
                            'type': 'keyword'
                        }
                    
                    structures.category_all_ids[kategori].update(keyword_ids)
    
    # === ADIM 5: İstatistikler ===
    for kategori in structures.category_phrase_ids.keys():
        structures.category_stats[kategori] = {
            'total_phrases': len(structures.category_phrase_ids[kategori]),
            'total_keywords': len(structures.category_keyword_ids[kategori]),
            'unique_ids': len(structures.category_all_ids[kategori])
        }
    
    # Şebeke Unsuru istatistikleri
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
    
    print(f"\n✅ Morfosemantik işleme tamamlandı!")
    print(f"   - {len(structures.unsur_to_categories)} şebeke unsuru")
    print(f"   - {len(structures.category_phrase_ids)} kategori")
    print(f"   - {len(seen_phrases)} unique phrase")
    print(f"   - {len(seen_keywords)} unique keyword")
    print(f"   - {len(structures.token_to_id)} morfosemantik token")
    
    # === ADIM 6: Morfosemantik dağılım istatistikleri ===
    print("\n📊 Morfosemantik İstatistikler:")
    
    pos_counts = defaultdict(int)
    
    for token in structures.id_to_token.values():
        pos_counts[token.pos] += 1
    
    print(f"   POS Dağılımı:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {pos}: {count}")
    
    return structures
