from typing import List, Set, Dict
from collections import defaultdict
import pandas as pd
import math


class IDBasedCategoryMatcher:
    def __init__(self, df, zemb, structures):
        self.zemb = zemb
        self.df = df
        
        # Structures'ı yükle
        self._initialize_from_structures(structures)
    
    def _initialize_from_structures(self, structures):
        """Structures'dan veri yapılarını yükle"""
        self.lemma2id = structures.lemma2id
        self.id2lemma = structures.id2lemma
        
        # ✨ Cause code ve şebeke unsuru mappings
        self.cause_code_to_unsur = structures.cause_code_to_unsur
        self.unsur_to_categories = structures.unsur_to_categories
        
        # ✨ YENİ: Çözüm Açıklama mappings
        self.çözüm_açıklama_to_kök_neden = structures.çözüm_açıklama_to_kök_neden
        self.çözüm_açıklama_to_kategori = structures.çözüm_açıklama_to_kategori
        self.çözüm_açıklama_to_unsur = structures.çözüm_açıklama_to_unsur
        self.çözüm_açıklama_to_cause_code = structures.çözüm_açıklama_to_cause_code
        
        # Global yapılar (eski)
        self.category_phrase_ids = structures.category_phrase_ids
        self.category_keyword_ids = structures.category_keyword_ids
        self.category_all_ids = structures.category_all_ids
        
        # ✨ Şebeke unsuru bazlı yapılar
        self.unsur_category_phrase_ids = structures.unsur_category_phrase_ids
        self.unsur_category_keyword_ids = structures.unsur_category_keyword_ids
        
        self.phrase_ids_to_info = structures.phrase_ids_to_info
        self.keyword_ids_to_info = structures.keyword_ids_to_info
        
        self.category_stats = structures.category_stats
        self.unsur_stats = structures.unsur_stats
        
    def tr_lower(self, text):
        """Türkçe karakterlere duyarlı lowercase"""
        if pd.isna(text):
            return ""
        replacements = {
            'I': 'ı', 'İ': 'i', 'Ğ': 'ğ', 'Ü': 'ü',
            'Ş': 'ş', 'Ö': 'ö', 'Ç': 'ç'
        }
        text = str(text)
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.lower()
    
    def get_lemma(self, word):
        """
        Kelimenin kökünü bul (olumsuzluk desteği ile)
        
        Returns:
            str: Kök (olumsuz ise olumsuz eki eklenmiş hali)
        """
        try:
            # analyze_sentence_full kullan (daha detaylı)
            resp = self.zemb.analyze_sentence_full(self.tr_lower(word))
            
            if resp and resp.results and len(resp.results) > 0:
                result = resp.results[0]
                
                # En iyi analizi al
                if result.best:
                    best = result.best
                    
                    # Lemma'yı al
                    if best.dictionaryItem and best.dictionaryItem.lemma:
                        base_lemma = self.tr_lower(best.dictionaryItem.lemma)
                        
                        # Morpheme'leri çıkar
                        morphemes = [m.morpheme for m in best.morphemes if m.morpheme]
                        
                        # Olumsuzluğu tespit et
                        is_negative = self._detect_true_negation(morphemes, best.pos)
                        
                        # Olumsuz ise olumsuz eki ekle
                        if is_negative:
                            return self._add_negative_suffix(base_lemma)
                        else:
                            return base_lemma
                    
                    # Fallback: lemmas varsa
                    elif best.lemmas:
                        return self.tr_lower(best.lemmas)
            
        except Exception as e:
            # Hata durumunda sessizce geç
            pass
        
        # Son çare: kelimeyi olduğu gibi döndür
        return self.tr_lower(word)
    
    def _detect_true_negation(self, morphemes, pos):
        """
        Gerçek olumsuzluğu tespit et
        
        Rules:
        1. Inf2 varsa → NEG değil (mastar: "oturma", "çalışma")
        2. Inf1 varsa → NEG değil (mastar: "oturmak")
        3. Neg + Imp → NEG değil (emir: "oturma!")
        4. Neg + Past/Pres/Fut → Gerçek olumsuz ("oturmadı", "oturmuyor")
        
        Args:
            morphemes: Morpheme listesi
            pos: Part of speech
        
        Returns:
            True sadece gerçek olumsuzsa
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
    
    def _add_negative_suffix(self, lemma):
        """
        Lemma'ya olumsuz eki ekle
        
        Args:
            lemma: Temel lemma (örn: "yanmak", "gelmek")
        
        Returns:
            Olumsuz lemma (örn: "yanmamak", "gelmemek")
        """
        # Türkçe olumsuz eki: -ma/-me (son ünlüye göre)
        if lemma.endswith("mak"):
            stem = lemma[:-3]  # "yan"
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
        
        elif lemma.endswith("mek"):
            stem = lemma[:-3]
            last_vowel = None
            for char in reversed(stem):
                if char in "aeıioöuü":
                    last_vowel = char
                    break
            
            if last_vowel in "aıou":
                return stem + "mamak"
            else:
                return stem + "memek"
        
        # Fallback: _NEG eki ekle
        return lemma + "_NEG"
    
    def text_to_lemma_ids(self, text):
        """
        Metni lemma ID'lerine çevir
        Returns: (id_list, lemma_list, original_words)
        """
        text = self.tr_lower(text)
        words = text.split()
        ids = []
        lemmas = []
        
        for word in words:
            
            lemma = self.get_lemma(word)
            
            if lemma != "unk":
                lemmas.append(lemma)
            
            # Lemma'yı ID'ye çevir
            if lemma != "unk" and lemma in self.lemma2id:
                ids.append(self.lemma2id[lemma])
        
        return ids, lemmas, words
    
    def find_matches(self, input_ids: List[int], input_id_set: Set[int]):
        """
        Input ID'leriyle eşleşen phrase, keyword ve TEK KELİMELERİ bul
        """
        exact_matches = []
        subset_matches = []
        partial_matches = []
        single_token_matches = []  # YENİ: Tek kelime eşleşmeleri
        
        # === 1. PHRASE MATCHING (Önceki gibi) ===
        for kategori, phrase_list in self.category_phrase_ids.items():
            for phrase_info in phrase_list:
                target_ids = phrase_info['ids']
                target_list = phrase_info['ids_list']
                
                # A) Exact match
                if target_ids == input_id_set:
                    exact_matches.append({
                        'kategori': kategori,
                        'type': 'phrase',
                        'match_type': 'exact',
                        'text': phrase_info['phrase'],
                        'matched_ids': list(target_ids),
                        'confidence': 1.0
                    })
                    continue
                
                # B) Subset match
                if target_ids.issubset(input_id_set):
                    is_sequential = self._check_sequential(input_ids, target_list)
                    confidence = 0.9 if is_sequential else 0.85
                    
                    subset_matches.append({
                        'kategori': kategori,
                        'type': 'phrase',
                        'match_type': 'subset_sequential' if is_sequential else 'subset',
                        'text': phrase_info['phrase'],
                        'matched_ids': list(target_ids),
                        'confidence': confidence,
                        'coverage': 1.0
                    })
                    continue
                
                # C) Partial match
                intersection = target_ids.intersection(input_id_set)
                if intersection:
                    coverage = len(intersection) / len(target_ids)
                    
                    if coverage >= 0.5:
                        confidence = coverage * 0.7
                        
                        partial_matches.append({
                            'kategori': kategori,
                            'type': 'phrase',
                            'match_type': 'partial',
                            'text': phrase_info['phrase'],
                            'matched_ids': list(intersection),
                            'missing_ids': list(target_ids - intersection),
                            'confidence': confidence,
                            'coverage': coverage
                        })
        
        # === 2. KEYWORD MATCHING (Önceki gibi) ===
        for kategori, keyword_list in self.category_keyword_ids.items():
            for keyword_info in keyword_list:
                target_ids = keyword_info['ids']
                
                if target_ids.issubset(input_id_set):
                    match_type = 'exact' if target_ids == input_id_set else 'subset'
                    confidence = 0.8 if match_type == 'exact' else 0.75
                    
                    subset_matches.append({
                        'kategori': kategori,
                        'type': 'keyword',
                        'match_type': match_type,
                        'text': keyword_info['keyword'],
                        'matched_ids': list(target_ids),
                        'confidence': confidence
                    })
        
        # === 3. SINGLE TOKEN MATCHING ===
        for input_id in input_ids:
            input_id_set_single = frozenset([input_id])
            
            # Bu ID hangi kategorilerde var?
            for kategori, all_ids in self.category_all_ids.items():
                if input_id in all_ids:
                    # Bu ID'nin hangi phrase/keyword'e ait olduğunu bul
                    matched_texts = []
                    
                    # Phrase'lerde ara
                    for phrase_info in self.category_phrase_ids[kategori]:
                        if input_id in phrase_info['ids']:
                            matched_texts.append({
                                'text': phrase_info['phrase'],
                                'type': 'phrase',
                                'token': self.id2lemma[input_id]
                            })
                    
                    # Keyword'lerde ara
                    for keyword_info in self.category_keyword_ids[kategori]:
                        if input_id in keyword_info['ids']:
                            matched_texts.append({
                                'text': keyword_info['keyword'],
                                'type': 'keyword',
                                'token': self.id2lemma[input_id]
                            })
                    
                    # Her eşleşen text için tek bir match oluştur
                    for text_info in matched_texts:
                        single_token_matches.append({
                            'kategori': kategori,
                            'type': 'single_token',
                            'match_type': 'single',
                            'text': text_info['text'],
                            'source_type': text_info['type'],
                            'token': text_info['token'],
                            'matched_ids': [input_id],
                            'confidence': 0.3
                        })
        
        return exact_matches, subset_matches, partial_matches, single_token_matches
    
    def _check_sequential(self, input_ids, target_list):
        """Target list'in input_ids içinde sıralı olup olmadığını kontrol et"""
        if not target_list:
            return False
        
        target_idx = 0
        for input_id in input_ids:
            if target_idx < len(target_list) and input_id == target_list[target_idx]:
                target_idx += 1
                if target_idx == len(target_list):
                    return True
        return False
    
    def calculate_category_scores_with_idf(self, exact, subset, partial, single_token):
        """
        Kategori skorlarını hesapla (IDF ile ağırlıklandırılmış)
        """
        category_scores = defaultdict(lambda: {
            'confidence': 0.0,
            'exact_count': 0,
            'subset_count': 0,
            'partial_count': 0,
            'single_token_count': 0,
            'total_matches': 0,
            'best_match_type': None,
            'match_details': []
        })
        
        # IDF değerlerini hesapla (her token kaç kategoride var?)
        token_category_count = defaultdict(int)
        for kategori in self.category_all_ids.keys():
            for token_id in self.category_all_ids[kategori]:
                token_category_count[token_id] += 1
        
        total_categories = len(self.category_all_ids)
        
        def get_idf_weight(token_id):
            """Token'in IDF ağırlığını hesapla (nadir tokenler daha yüksek)"""
            if token_id not in token_category_count:
                return 1.0
            
            doc_freq = token_category_count[token_id]
            idf = math.log((total_categories + 1) / (doc_freq + 1)) + 1
            return idf
        
        # Exact matches (en yüksek ağırlık)
        for match in exact:
            cat = match['kategori']
            
            # IDF ağırlıklı confidence
            token_weights = [get_idf_weight(tid) for tid in match['matched_ids']]
            avg_idf = sum(token_weights) / len(token_weights) if token_weights else 1.0
            weighted_confidence = match['confidence'] * avg_idf
            
            category_scores[cat]['confidence'] += weighted_confidence
            category_scores[cat]['exact_count'] += 1
            category_scores[cat]['total_matches'] += 1
            category_scores[cat]['match_details'].append({
                'type': 'exact',
                'text': match['text'],
                'confidence': match['confidence'],
                'idf_weight': avg_idf
            })
            
            if not category_scores[cat]['best_match_type']:
                category_scores[cat]['best_match_type'] = 'exact'
        
        # Subset matches
        for match in subset:
            cat = match['kategori']
            
            token_weights = [get_idf_weight(tid) for tid in match['matched_ids']]
            avg_idf = sum(token_weights) / len(token_weights) if token_weights else 1.0
            weighted_confidence = match['confidence'] * avg_idf * 0.9
            
            category_scores[cat]['confidence'] += weighted_confidence
            category_scores[cat]['subset_count'] += 1
            category_scores[cat]['total_matches'] += 1
            category_scores[cat]['match_details'].append({
                'type': 'subset',
                'text': match['text'],
                'confidence': match['confidence'],
                'idf_weight': avg_idf
            })
            
            if not category_scores[cat]['best_match_type']:
                category_scores[cat]['best_match_type'] = 'subset'
        
        # Partial matches
        for match in partial:
            cat = match['kategori']
            
            token_weights = [get_idf_weight(tid) for tid in match['matched_ids']]
            avg_idf = sum(token_weights) / len(token_weights) if token_weights else 1.0
            weighted_confidence = match['confidence'] * avg_idf * 0.7
            
            category_scores[cat]['confidence'] += weighted_confidence
            category_scores[cat]['partial_count'] += 1
            category_scores[cat]['total_matches'] += 1
            category_scores[cat]['match_details'].append({
                'type': 'partial',
                'text': match['text'],
                'confidence': match['confidence'],
                'coverage': match.get('coverage', 0),
                'idf_weight': avg_idf
            })
            
            if not category_scores[cat]['best_match_type']:
                category_scores[cat]['best_match_type'] = 'partial'
        
        # Single token matches (en düşük ağırlık ama IDF önemli)
        for match in single_token:
            cat = match['kategori']
            
            token_id = match['matched_ids'][0]
            idf_weight = get_idf_weight(token_id)
            
            # Nadir tokenler için daha yüksek confidence
            weighted_confidence = match['confidence'] * idf_weight * 0.5
            
            category_scores[cat]['confidence'] += weighted_confidence
            category_scores[cat]['single_token_count'] += 1
            category_scores[cat]['total_matches'] += 1
            category_scores[cat]['match_details'].append({
                'type': 'single_token',
                'token': match['token'],
                'text': match['text'],
                'confidence': match['confidence'],
                'idf_weight': idf_weight
            })
            
            if not category_scores[cat]['best_match_type']:
                category_scores[cat]['best_match_type'] = 'single_token'
        
        return dict(category_scores)
    
    def normalize_scores(self, category_scores):
        """Skorları [0,1] aralığında normalize et"""
        if not category_scores:
            return {}
        
        max_score = max(info['confidence'] for info in category_scores.values())
        
        if max_score == 0:
            return category_scores
        
        for cat in category_scores:
            category_scores[cat]['normalized_confidence'] = (
                category_scores[cat]['confidence'] / max_score
            )
        
        return category_scores
    
    def _format_predictions(self, sorted_categories):
        """Tahminleri formatla"""
        predictions = []
        
        for cat, info in sorted_categories:
            predictions.append({
                'kategori': cat,
                'confidence': round(info['normalized_confidence'], 4),
                'raw_confidence': round(info['confidence'], 4),
                'match_summary': {
                    'exact': info['exact_count'],
                    'subset': info['subset_count'],
                    'partial': info['partial_count'],
                    'single_token': info['single_token_count'],
                    'total': info['total_matches']
                },
                'best_match_type': info['best_match_type'],
                'match_details': info['match_details'][:5]  # İlk 5 detayı göster
            })
        
        return predictions

    def predict(
        self,
        user_input: str,
        cause_code: str = None,
        çözüm_açıklama: str = None,  # ✨ YENİ PARAMETRE
        top_k: int = 5,
        min_confidence: float = 0.1
    ):
        """
        ✨ YENİ: Çözüm Açıklama öncelikli tahmin sistemi
        
        Akış:
        1. Çözüm Açıklama → Kök Neden var mı? → Varsa döndür
        2. Çözüm Açıklama → Kategori var mı? → Varsa döndür
        3. Çözüm Açıklama & Cause Code → Şebeke Unsurları karşılaştır
        4. Arama yap (filtered veya global)
        
        Args:
            user_input: Kullanıcı metni
            cause_code: Cause code (opsiyonel)
            çözüm_açıklama: Çözüm açıklama metni (opsiyonel)
            top_k: Döndürülecek tahmin sayısı
            min_confidence: Minimum güven skoru
        """
        
        print("\n" + "="*60)
        print("🔍 TAHMİN SÜRECİ BAŞLIYOR")
        print("="*60)
        
        # === AŞAMA 1: Çözüm Açıklama → Kök Neden Kontrolü ===
        if çözüm_açıklama:
            print(f"\n📋 Çözüm Açıklama: {çözüm_açıklama}")
            
            # 1.1. Kök Neden var mı?
            kök_neden = self.çözüm_açıklama_to_kök_neden.get(çözüm_açıklama)
            if kök_neden and kök_neden != "-":
                print(f"✅ KÖK NEDEN BULUNDU: {kök_neden}")
                return {
                    'input': user_input,
                    'çözüm_açıklama': çözüm_açıklama,
                    'kök_neden': kök_neden,
                    'method': 'direct_from_kök_neden',
                    'predictions': [{
                        'kategori': kök_neden,
                        'confidence': 1.0,
                        'source': 'Çözüm Açıklama → Kök Neden mapping',
                        'match_summary': {'direct_mapping': 1}
                    }]
                }
            else:
                print("❌ Kök Neden bulunamadı")
            
            # 1.2. Kategori var mı?
            kategori = self.çözüm_açıklama_to_kategori.get(çözüm_açıklama)
            if kategori and kategori != "-":
                print(f"✅ KATEGORİ BULUNDU: {kategori}")
                return {
                    'input': user_input,
                    'çözüm_açıklama': çözüm_açıklama,
                    'kategori': kategori,
                    'method': 'direct_from_kategori',
                    'predictions': [{
                        'kategori': kategori,
                        'confidence': 1.0,
                        'source': 'Çözüm Açıklama → Kategori mapping',
                        'match_summary': {'direct_mapping': 1}
                    }]
                }
            else:
                print("❌ Kategori bulunamadı")
            
            # 1.3. Çözüm Açıklama'dan Cause Code al
            cause_code_from_cozum = self.çözüm_açıklama_to_cause_code.get(çözüm_açıklama)
            if cause_code_from_cozum:
                print(f"✅ Çözüm Açıklama'dan Cause Code bulundu: {cause_code_from_cozum}")
                # Parametre olarak gelen cause_code'u override et
                if cause_code and cause_code != cause_code_from_cozum:
                    print(f"⚠️  Uyarı: Parametre cause_code ({cause_code}) farklı, "
                          f"Çözüm Açıklama'dan gelen ({cause_code_from_cozum}) kullanılıyor")
                cause_code = cause_code_from_cozum
            else:
                print("❌ Çözüm Açıklama'dan Cause Code bulunamadı")
        
        # === AŞAMA 2: Cause Code ve Şebeke Unsuru İşleme ===
        print("\n" + "-"*60)
        print("⚙️  AŞAMA 2: Cause Code ve Şebeke Unsuru Kontrolü")
        print("-"*60)
        
        if not cause_code:
            print("❌ Cause Code yok → GLOBAL ARAMA yapılacak")
            search_all_categories = True
            sebeke_unsuru = None
            relevant_categories = set(self.category_phrase_ids.keys())
        else:
            print(f"✅ Cause Code: {cause_code}")
            
            # 2.1. Cause Code → Şebeke Unsuru (cause_code_df'den)
            sebeke_unsuru_from_cc = self.cause_code_to_unsur.get(cause_code)
            print(f"   Cause Code → Şebeke Unsuru: {sebeke_unsuru_from_cc or 'BULUNAMADI'}")
            
            # 2.2. Çözüm Açıklama → Şebeke Unsuru
            sebeke_unsuru_from_cozum = None
            if çözüm_açıklama:
                sebeke_unsuru_from_cozum = self.çözüm_açıklama_to_unsur.get(çözüm_açıklama)
                print(f"   Çözüm Açıklama → Şebeke Unsuru: {sebeke_unsuru_from_cozum or 'BULUNAMADI'}")
            
            # 2.3. Şebeke Unsurlarını karşılaştır
            if sebeke_unsuru_from_cc and sebeke_unsuru_from_cozum:
                if sebeke_unsuru_from_cc == sebeke_unsuru_from_cozum:
                    print(f"✅ Şebeke Unsurları AYNI: {sebeke_unsuru_from_cc}")
                    print(f"   → FİLTRELİ ARAMA yapılacak (sadece {sebeke_unsuru_from_cc} altında)")
                    sebeke_unsuru = sebeke_unsuru_from_cc
                    search_all_categories = False
                else:
                    print(f"⚠️  Şebeke Unsurları FARKLI:")
                    print(f"   - Cause Code'dan: {sebeke_unsuru_from_cc}")
                    print(f"   - Çözüm Açıklama'dan: {sebeke_unsuru_from_cozum}")
                    print(f"   → GLOBAL ARAMA yapılacak (tüm kategorilerde)")
                    sebeke_unsuru = None
                    search_all_categories = True
            elif sebeke_unsuru_from_cc:
                print(f"✅ Cause Code'dan Şebeke Unsuru: {sebeke_unsuru_from_cc}")
                print(f"   → FİLTRELİ ARAMA yapılacak")
                sebeke_unsuru = sebeke_unsuru_from_cc
                search_all_categories = False
            elif sebeke_unsuru_from_cozum:
                print(f"✅ Çözüm Açıklama'dan Şebeke Unsuru: {sebeke_unsuru_from_cozum}")
                print(f"   → FİLTRELİ ARAMA yapılacak")
                sebeke_unsuru = sebeke_unsuru_from_cozum
                search_all_categories = False
            else:
                print("❌ Şebeke Unsuru bulunamadı → GLOBAL ARAMA yapılacak")
                sebeke_unsuru = None
                search_all_categories = True
            
            # 2.4. İlgili kategorileri belirle
            if not search_all_categories:
                relevant_categories = self.unsur_to_categories.get(sebeke_unsuru, set())
                
                if not relevant_categories:
                    print(f"⚠️  Uyarı: '{sebeke_unsuru}' için kategori bulunamadı → GLOBAL ARAMA'ya geçiliyor")
                    search_all_categories = True
                    relevant_categories = set(self.category_phrase_ids.keys())
            else:
                relevant_categories = set(self.category_phrase_ids.keys())
        
        # === AŞAMA 3: Text İşleme ===
        print("\n" + "-"*60)
        print("⚙️  AŞAMA 3: Metin Analizi")
        print("-"*60)
        
        print(f"\n🔍 Arama Kapsamı:")
        print(f"   Cause Code: {cause_code or 'YOK'}")
        print(f"   Şebeke Unsuru: {sebeke_unsuru or 'YOK'}")
        print(f"   Arama Modu: {'🌍 GLOBAL (Tüm kategoriler)' if search_all_categories else f'🎯 FİLTRELİ ({sebeke_unsuru})'}")
        print(f"   İlgili Kategoriler: {len(relevant_categories)} adet")
        if not search_all_categories:
            print(f"   Kategoriler: {list(relevant_categories)[:10]}{'...' if len(relevant_categories) > 10 else ''}")
        
        # 3.1. Input'u ID'lere çevir
        input_ids, input_lemmas, input_words = self.text_to_lemma_ids(user_input)
        input_id_set = set(input_ids)
        
        print(f"\n📝 Input: {user_input}")
        print(f"   Kelimeler: {input_words}")
        print(f"   Kökler: {input_lemmas}")
        print(f"   ID'ler: {input_ids}")
        
        if not input_ids:
            return {
                'input': user_input,
                'çözüm_açıklama': çözüm_açıklama,
                'cause_code': cause_code,
                'sebeke_unsuru': sebeke_unsuru,
                'search_mode': 'global' if search_all_categories else 'filtered',
                'error': 'Geçerli kelime bulunamadı',
                'predictions': []
            }
        
        # === AŞAMA 4: Arama ===
        print("\n" + "-"*60)
        print("⚙️  AŞAMA 4: Eşleşme Arama")
        print("-"*60)
        
        if search_all_categories:
            # TÜM KATEGORİLERDE ARA
            exact, subset, partial, single_token = self._find_matches_global(
                input_ids, 
                input_id_set
            )
        else:
            # SADECE İLGİLİ KATEGORİLERDE ARA
            exact, subset, partial, single_token = self._find_matches_filtered(
                input_ids, 
                input_id_set, 
                sebeke_unsuru,
                relevant_categories
            )
        
        print(f"\n📊 Eşleşme Sonuçları:")
        print(f"   Exact matches: {len(exact)}")
        print(f"   Subset matches: {len(subset)}")
        print(f"   Partial matches: {len(partial)}")
        print(f"   Single token matches: {len(single_token)}")
        
        # === AŞAMA 5: Skorlama ===
        print("\n" + "-"*60)
        print("⚙️  AŞAMA 5: Skorlama ve Sıralama")
        print("-"*60)
        
        category_scores = self.calculate_category_scores_with_idf(
            exact, subset, partial, single_token
        )
        
        category_scores = self.normalize_scores(category_scores)
        
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1]['normalized_confidence'],
            reverse=True
        )
        
        sorted_categories = [
            (cat, info) for cat, info in sorted_categories
            if info['normalized_confidence'] >= min_confidence
        ][:top_k]
        
        print(f"\n✅ Top {len(sorted_categories)} Tahmin:")
        for i, (cat, info) in enumerate(sorted_categories, 1):
            print(f"   {i}. {cat}: {info['normalized_confidence']:.4f} "
                  f"(E:{info['exact_count']}, S:{info['subset_count']}, "
                  f"P:{info['partial_count']}, ST:{info['single_token_count']})")
        
        # === AŞAMA 6: Çıktı ===
        predictions = self._format_predictions(sorted_categories)
        
        return {
            'input': user_input,
            'çözüm_açıklama': çözüm_açıklama,
            'cause_code': cause_code,
            'sebeke_unsuru': sebeke_unsuru,
            'search_mode': 'global' if search_all_categories else 'filtered',
            'method': 'text_matching',
            'input_ids': input_ids,
            'input_lemmas': input_lemmas,
            'input_words': input_words,
            'search_scope': {
                'total_categories': len(relevant_categories),
                'categories': list(relevant_categories)[:10] if not search_all_categories else ['TÜM KATEGORİLER']
            },
            'total_exact_matches': len(exact),
            'total_subset_matches': len(subset),
            'total_partial_matches': len(partial),
            'total_single_token_matches': len(single_token),
            'predictions': predictions
        }

    def _find_matches_global(
        self, 
        input_ids: List[int], 
        input_id_set: Set[int]
    ):
        """
        TÜM kategorilerde ara (şebeke unsuru yok ise)
        """
        exact_matches = []
        subset_matches = []
        partial_matches = []
        single_token_matches = []
        
        # TÜM kategorilerde ara
        for kategori, phrase_list in self.category_phrase_ids.items():
            # Phrase matching
            for phrase_info in phrase_list:
                match = self._check_phrase_match(input_ids, input_id_set, phrase_info, kategori)
                if match:
                    if match['match_type'] == 'exact':
                        exact_matches.append(match)
                    elif 'subset' in match['match_type']:
                        subset_matches.append(match)
                    else:
                        partial_matches.append(match)
        
        # Keyword matching
        for kategori, keyword_list in self.category_keyword_ids.items():
            for keyword_info in keyword_list:
                match = self._check_keyword_match(input_id_set, keyword_info, kategori)
                if match:
                    subset_matches.append(match)
        
        # Single token matching
        for kategori in self.category_phrase_ids.keys():
            single_tokens = self._find_single_tokens(input_ids, kategori)
            single_token_matches.extend(single_tokens)
        
        return exact_matches, subset_matches, partial_matches, single_token_matches
    
    def _find_matches_filtered(
        self, 
        input_ids: List[int], 
        input_id_set: Set[int],
        sebeke_unsuru: str,
        relevant_categories: Set[str]
    ):
        """
        Sadece ilgili kategorilerde ara (şebeke unsuru belli ise)
        """
        exact_matches = []
        subset_matches = []
        partial_matches = []
        single_token_matches = []
        
        # Sadece bu şebeke unsurundaki kategorilerde ara
        for kategori in relevant_categories:
            # Phrase matching
            if kategori in self.unsur_category_phrase_ids.get(sebeke_unsuru, {}):
                for phrase_info in self.unsur_category_phrase_ids[sebeke_unsuru][kategori]:
                    match = self._check_phrase_match(input_ids, input_id_set, phrase_info, kategori)
                    if match:
                        if match['match_type'] == 'exact':
                            exact_matches.append(match)
                        elif 'subset' in match['match_type']:
                            subset_matches.append(match)
                        else:
                            partial_matches.append(match)
            
            # Keyword matching
            if kategori in self.unsur_category_keyword_ids.get(sebeke_unsuru, {}):
                for keyword_info in self.unsur_category_keyword_ids[sebeke_unsuru][kategori]:
                    match = self._check_keyword_match(input_id_set, keyword_info, kategori)
                    if match:
                        subset_matches.append(match)
            
            # Single token matching
            single_tokens = self._find_single_tokens(input_ids, kategori)
            single_token_matches.extend(single_tokens)
        
        return exact_matches, subset_matches, partial_matches, single_token_matches
    
    def _check_phrase_match(self, input_ids, input_id_set, phrase_info, kategori):
        """Phrase eşleşmesi kontrol et"""
        target_ids = phrase_info['ids']
        target_list = phrase_info['ids_list']
        
        # A) Exact match
        if target_ids == input_id_set:
            return {
                'kategori': kategori,
                'type': 'phrase',
                'match_type': 'exact',
                'text': phrase_info['phrase'],
                'matched_ids': list(target_ids),
                'confidence': 1.0
            }
        
        # B) Subset match
        if target_ids.issubset(input_id_set):
            is_sequential = self._check_sequential(input_ids, target_list)
            confidence = 0.9 if is_sequential else 0.85
            
            return {
                'kategori': kategori,
                'type': 'phrase',
                'match_type': 'subset_sequential' if is_sequential else 'subset',
                'text': phrase_info['phrase'],
                'matched_ids': list(target_ids),
                'confidence': confidence,
                'coverage': 1.0
            }
        
        # C) Partial match
        intersection = target_ids.intersection(input_id_set)
        if intersection:
            coverage = len(intersection) / len(target_ids)
            
            if coverage >= 0.5:
                confidence = coverage * 0.7
                
                return {
                    'kategori': kategori,
                    'type': 'phrase',
                    'match_type': 'partial',
                    'text': phrase_info['phrase'],
                    'matched_ids': list(intersection),
                    'missing_ids': list(target_ids - intersection),
                    'confidence': confidence,
                    'coverage': coverage
                }
        
        return None
    
    def _check_keyword_match(self, input_id_set, keyword_info, kategori):
        """Keyword eşleşmesi kontrol et"""
        target_ids = keyword_info['ids']
        
        if target_ids.issubset(input_id_set):
            match_type = 'exact' if target_ids == input_id_set else 'subset'
            confidence = 0.8 if match_type == 'exact' else 0.75
            
            return {
                'kategori': kategori,
                'type': 'keyword',
                'match_type': match_type,
                'text': keyword_info['keyword'],
                'matched_ids': list(target_ids),
                'confidence': confidence
            }
        
        return None
    
    def _find_single_tokens(self, input_ids, kategori):
        """Tek token eşleşmelerini bul"""
        single_token_matches = []
        
        for input_id in input_ids:
            if input_id in self.category_all_ids.get(kategori, set()):
                # Bu ID'nin hangi phrase/keyword'e ait olduğunu bul
                matched_texts = []
                
                # Phrase'lerde ara
                for phrase_info in self.category_phrase_ids.get(kategori, []):
                    if input_id in phrase_info['ids']:
                        matched_texts.append({
                            'text': phrase_info['phrase'],
                            'type': 'phrase',
                            'token': self.id2lemma[input_id]
                        })
                
                # Keyword'lerde ara
                for keyword_info in self.category_keyword_ids.get(kategori, []):
                    if input_id in keyword_info['ids']:
                        matched_texts.append({
                            'text': keyword_info['keyword'],
                            'type': 'keyword',
                            'token': self.id2lemma[input_id]
                        })
                
                # Her eşleşen text için match oluştur
                for text_info in matched_texts:
                    single_token_matches.append({
                        'kategori': kategori,
                        'type': 'single_token',
                        'match_type': 'single',
                        'text': text_info['text'],
                        'source_type': text_info['type'],
                        'token': text_info['token'],
                        'matched_ids': [input_id],
                        'confidence': 0.3
                    })
        
        return single_token_matches
    
    def debug_info(self, kategori: str = None):
        """Debug için kategori bilgilerini göster"""
        if kategori:
            print(f"\n📊 KATEGORİ: {kategori}")
            print(f"   Toplam Phrases: {self.category_stats[kategori]['total_phrases']}")
            print(f"   Toplam Keywords: {self.category_stats[kategori]['total_keywords']}")
            print(f"   Unique IDs: {self.category_stats[kategori]['unique_ids']}")
            
            print(f"\n   Phrase ID Setleri:")
            for p in self.category_phrase_ids[kategori][:5]:  # İlk 5'i göster
                print(f"      {p['phrase']}: {p['ids_list']}")
        else:
            print("\n📊 GENEL İSTATİSTİKLER:")
            print(f"   Toplam Kategori: {len(self.category_stats)}")
            print(f"   Toplam Lemma: {len(self.lemma2id)}")
            print(f"\n   Kategoriler:")
            for cat, stats in list(self.category_stats.items())[:10]:
                print(f"      {cat}: {stats['total_phrases']} phrases, {stats['total_keywords']} keywords")