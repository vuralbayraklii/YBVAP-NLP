"""
Morfosemantik Arıza Analiz Sistemi v3

Bu modül, arıza metinlerini morfosemantik analiz ederek yapılandırılmış bir
veri yapısı oluşturur. Temel özellikler:

1. Morfosemantik Token: Lemma + POS + Olumsuzluk bilgisi
2. ArızaStructures: Tüm mapping'leri tutan ana veri yapısı
3. Akıllı olumsuzluk tespiti (mastar ve emir formlarını ayırt eder)
4. Çözüm Açıklama → Kategori/Unsur/Kök Neden/Cause Code mapping'leri

Mapping Formatı:
- lemma2id: Sadece kökler kaydedilir (örn: "yanmak", "haberleşmemek")
- Olumsuz tokenler olumsuz eki ile kaydedilir (örn: "haberleşmemek")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from collections import defaultdict
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
    token_to_id: Dict[str, int] = field(default_factory=dict)  # "kapamak_Verb_NEG" → ID (internal)
    id_to_token: Dict[int, MorphosemanticToken] = field(default_factory=dict)
    
    # Ana mapping: Sadece kökler (olumsuz ise olumsuz haliyle)
    lemma2id: Dict[str, int] = field(default_factory=dict)  # "yanmak" → ID, "haberleşmemek" → ID
    id2lemma: Dict[int, str] = field(default_factory=dict)  # ID → "yanmak", ID → "haberleşmemek"
    
    # Çözüm Açıklama --> Kök Neden
    çözüm_açıklama_to_kök_neden: Dict[str, str] = field(default_factory=dict) # YENİ EKLENDİ
    
    # Çözüm Açıklama --> Kategori
    çözüm_açıklama_to_kategori: Dict[str, str] = field(default_factory=dict) # YENİ EKLENDİ
    
    # Çözüm Açıklama --> Şebeke Unsuru
    çözüm_açıklama_to_unsur: Dict[str, str] = field(default_factory=dict) # YENİ EKLENDİ
    
    # Çözüm Açıklama --> Cause Code
    çözüm_açıklama_to_cause_code: Dict[str, str] = field(default_factory=dict) # YENİ EKLENDİ
    
    # Cause Code → Şebeke Unsuru mapping
    cause_code_to_unsur: Dict[str, str] = field(default_factory=dict)
    
    # Cause Code mappings (YENİ EKLENENLER)
    cause_code_to_unsur: Dict[str, str] = field(default_factory=dict)
    cause_code_to_kategori: Dict[str, str] = field(default_factory=dict)  # YENİ
    cause_code_to_kök_neden: Dict[str, str] = field(default_factory=dict)  # YENİ
 
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


def tr_lower(text: str) -> str:
    """
    Türkçe karakterleri koruyarak küçük harfe çevir
    
    Args:
        text: Input string
    
    Returns:
        Lowercase string
    """
    if pd.isna(text):
        return ""
    
    # Türkçe karakter dönüşümleri
    replacements = {
        'İ': 'i',
        'I': 'ı',
        'Ğ': 'ğ',
        'Ü': 'ü',
        'Ş': 'ş',
        'Ö': 'ö',
        'Ç': 'ç'
    }
    
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result.lower()


def fuzzy_match(word: str, dictionary: List[str], threshold: int = 2) -> Optional[str]:
    """
    Fuzzy matching ile en yakın kelimeyi bul
    
    Args:
        word: Aranacak kelime
        dictionary: Kelime listesi
        threshold: Maksimum Levenshtein mesafesi
    
    Returns:
        En yakın kelime veya None
    """
    try:
        from Levenshtein import distance
    except ImportError:
        # Levenshtein yoksa basit bir fallback
        return None
    
    best_match = None
    best_distance = float('inf')
    
    for candidate in dictionary:
        d = distance(word.lower(), candidate.lower())
        if d < best_distance and d <= threshold:
            best_distance = d
            best_match = candidate
    
    return best_match


def is_valid_token(token: MorphosemanticToken) -> bool:
    """
    Token'in geçerli olup olmadığını kontrol et
    
    Args:
        token: MorphosemanticToken
    
    Returns:
        True if valid, False otherwise
    """
    # Noktalama ve sayıları filtrele
    if token.pos in ["Punc", "Num", "Unknown"]:
        return False
    
    # Çok kısa lemma'ları filtrele
    if len(token.lemma) < 2:
        return False
    
    return True


def get_morphosemantic_id(
    token: MorphosemanticToken, 
    structures: ArızaStructures, 
    auto_create: bool = True
) -> Optional[int]:
    """
    Token için ID al (varsa döndür, yoksa oluştur)
    
    ÖNEMLI: Sadece display_lemma kullanılır (POS bilgisi ID'ye dahil değil)
    - "yanmak" (Noun) ve "yanmak" (Verb) → Aynı ID
    - "haberleşmek" (olumlu) ve "haberleşmemek" (olumsuz) → Farklı ID
    
    Args:
        token: MorphosemanticToken
        structures: ArızaStructures instance
        auto_create: Yeni ID oluşturulsun mu?
    
    Returns:
        Token ID veya None
    """
    # Display lemma'yı al (olumsuz ise olumsuz haliyle)
    display_lemma = token.to_display_lemma()
    
    # Önce lemma2id'de var mı bak
    if display_lemma in structures.lemma2id:
        existing_id = structures.lemma2id[display_lemma]
        
        # Internal mapping'i de güncelle (aynı lemma farklı POS'larla gelebilir)
        token_key = token.to_key()
        if token_key not in structures.token_to_id:
            structures.token_to_id[token_key] = existing_id
            # id_to_token'da sadece ilk görüleni sakla
            if existing_id not in structures.id_to_token:
                structures.id_to_token[existing_id] = token
        
        return existing_id
    
    # Yoksa ve auto_create açıksa oluştur
    if auto_create:
        # Yeni ID oluştur
        new_id = len(structures.lemma2id)
        
        # Kaydet - SADECE display_lemma kullan
        structures.lemma2id[display_lemma] = new_id
        structures.id2lemma[new_id] = display_lemma
        
        # Internal mapping'i de kaydet (tam bilgi için)
        token_key = token.to_key()
        structures.token_to_id[token_key] = new_id
        structures.id_to_token[new_id] = token
        
        return new_id
    
    return None


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
        print(f"⚠️ Analiz hatası: {text} → {e}")
        return []


def build_morphosemantic_structures_v3(
    arizalar_df: pd.DataFrame,
    zemb,
    dokunma=None,
    çözüm_açıklama_info_df: pd.DataFrame = None,
    cause_code_info_df: pd.DataFrame = None  # YENİ PARAMETRE
) -> ArızaStructures:
    """
    Morfosemantik analiz + ArızaStructures oluştur
    
    Args:
        arizalar_df: Ana arıza DataFrame'i (Key Phrase, Keywords, Kategori, vs)
        zemb: Zemberek analyzer
        cause_code_df_path: Cause Code → Şebeke Unsuru mapping dosyası (Excel/CSV)
        dokunma: Fuzzy matching için dictionary (opsiyonel)
        çözüm_açıklama_info_df: Çözüm Açıklama bilgilerini içeren DataFrame
            Kolonlar: ['Çözüm Açıklama', 'Şebeke Unsuru', 'Arıza Kategorisi', 'Arıza Kök-Neden']
        çözüm_açıklama_to_cause_code_df: Çözüm Açıklama → Cause Code mapping DataFrame
            Kolonlar: ['cause code', 'Çözüm Açıklama']
    
    Returns:
        ArızaStructures instance
    """
    
    print("🚀 Morfosemantik işleme başlıyor...")
    structures = ArızaStructures()
    
    if dokunma is None:
        dokunma = {}

    # === YENİ: Cause Code bilgilerini yükle ===
    if cause_code_info_df is not None:
        print("   📋 Cause Code bilgileri yükleniyor...")
        
        for _, row in cause_code_info_df.iterrows():
            if 'Cause Code' in cause_code_info_df.columns:
                cause_code = row['Cause Code']
            elif 'cause code' in cause_code_info_df.columns:
                cause_code = row['cause code']
            else:
                # Fallback: case-insensitive column search for "cause code"
                matched_col = None
                for col in cause_code_info_df.columns:
                    if str(col).strip().lower() == 'cause code':
                        matched_col = col
                        break
                cause_code = row[matched_col] if matched_col is not None else None
            
            if pd.notna(cause_code):
                # Cause Code → Şebeke Unsuru
                if pd.notna(row['Şebeke Unsuru']):
                    structures.cause_code_to_unsur[cause_code] = row['Şebeke Unsuru']
                
                # Cause Code → Arıza Kategorisi
                if pd.notna(row['Arıza Kategorisi']):
                    if not hasattr(structures, 'cause_code_to_kategori'):
                        structures.cause_code_to_kategori = {}
                    structures.cause_code_to_kategori[cause_code] = row['Arıza Kategorisi']
                
                # Cause Code → Arıza Kök-Neden
                if pd.notna(row['Arıza Kök-Neden']):
                    if not hasattr(structures, 'cause_code_to_kök_neden'):
                        structures.cause_code_to_kök_neden = {}
                    structures.cause_code_to_kök_neden[cause_code] = row['Arıza Kök-Neden']
        
        print(f"   ✅ {len(structures.cause_code_to_unsur)} Cause Code → Şebeke Unsuru")
        if hasattr(structures, 'cause_code_to_kategori'):
            print(f"   ✅ {len(structures.cause_code_to_kategori)} Cause Code → Kategori")
        if hasattr(structures, 'cause_code_to_kök_neden'):
            print(f"   ✅ {len(structures.cause_code_to_kök_neden)} Cause Code → Kök Neden")
    
    # === Çözüm Açıklama mapping'lerini doldur ===
    if çözüm_açıklama_info_df is not None:
        print("   📋 Çözüm Açıklama bilgileri yükleniyor...")
        
        for _, row in çözüm_açıklama_info_df.iterrows():
            çözüm_açıklama = row['Çözüm Açıklama']
            
            if pd.notna(çözüm_açıklama):
                # Çözüm Açıklama → Şebeke Unsuru
                if pd.notna(row['Şebeke Unsuru']):
                    structures.çözüm_açıklama_to_unsur[çözüm_açıklama] = row['Şebeke Unsuru']
                
                # Çözüm Açıklama → Arıza Kategorisi
                if pd.notna(row['Arıza Kategorisi']):
                    structures.çözüm_açıklama_to_kategori[çözüm_açıklama] = row['Arıza Kategorisi']
                
                # Çözüm Açıklama → Arıza Kök-Neden
                if pd.notna(row['Arıza Kök-Neden']):
                    structures.çözüm_açıklama_to_kök_neden[çözüm_açıklama] = row['Arıza Kök-Neden']
        
        print(f"   ✅ {len(structures.çözüm_açıklama_to_unsur)} Çözüm Açıklama → Şebeke Unsuru")
        print(f"   ✅ {len(structures.çözüm_açıklama_to_kategori)} Çözüm Açıklama → Kategori")
        print(f"   ✅ {len(structures.çözüm_açıklama_to_kök_neden)} Çözüm Açıklama → Kök Neden")
    
    # === YENİ EKLEME: Çözüm Açıklama mapping'lerini doldur ===
    if çözüm_açıklama_info_df is not None:
        print("   📋 Çözüm Açıklama bilgileri yükleniyor...")
        
        for _, row in çözüm_açıklama_info_df.iterrows():
            çözüm_açıklama = row['Çözüm Açıklama']
            
            if pd.notna(çözüm_açıklama):
                # Çözüm Açıklama → Şebeke Unsuru
                if pd.notna(row['Şebeke Unsuru']):
                    structures.çözüm_açıklama_to_unsur[çözüm_açıklama] = row['Şebeke Unsuru']
                
                # Çözüm Açıklama → Arıza Kategorisi
                if pd.notna(row['Arıza Kategorisi']):
                    structures.çözüm_açıklama_to_kategori[çözüm_açıklama] = row['Arıza Kategorisi']
                
                # Çözüm Açıklama → Arıza Kök-Neden
                if pd.notna(row['Arıza Kök-Neden']):
                    structures.çözüm_açıklama_to_kök_neden[çözüm_açıklama] = row['Arıza Kök-Neden']
        
        print(f"   ✅ {len(structures.çözüm_açıklama_to_unsur)} Çözüm Açıklama → Şebeke Unsuru")
        print(f"   ✅ {len(structures.çözüm_açıklama_to_kategori)} Çözüm Açıklama → Kategori")
        print(f"   ✅ {len(structures.çözüm_açıklama_to_kök_neden)} Çözüm Açıklama → Kök Neden")

    # === ADIM 1: Tüm unique phrase/keyword'leri topla ===
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
    
    print(f"   ✅ {len(structures.lemma2id)} benzersiz kök (lemma) bulundu")
    print(f"   ℹ️  {len(structures.token_to_id)} farklı POS/NEG kombinasyonu")
    
    # İstatistikler
    neg_count = sum(1 for lemma in structures.lemma2id.keys() 
                    if any(neg_marker in lemma for neg_marker in ['memek', 'mamak', '_NEG']))
    
    print(f"   📊 Olumsuz kökler: {neg_count}")
    print(f"   📊 Olumlu kökler: {len(structures.lemma2id) - neg_count}")
    
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
    print(f"   - {len(structures.lemma2id)} benzersiz kök (lemma)")
    
    # === ADIM 6: Morfosemantik dağılım istatistikleri ===
    print("\n📊 Kök (Lemma) Özeti:")
    print(f"   Toplam benzersiz kök: {len(structures.lemma2id)}")
    
    # Olumsuz kök sayısı
    neg_count = sum(1 for lemma in structures.lemma2id.keys() 
                    if any(neg_marker in lemma for neg_marker in ['memek', 'mamak', '_NEG']))
    print(f"   Olumsuz kökler: {neg_count}")
    print(f"   Olumlu kökler: {len(structures.lemma2id) - neg_count}")
    
    # POS dağılımı (internal mapping'den)
    print("\n📊 POS Dağılımı (internal):")
    pos_counts = defaultdict(int)
    for token in structures.id_to_token.values():
        pos_counts[token.pos] += 1
    
    for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"     - {pos}: {count}")
    
    return structures
