# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:50:20 2025

@author: vural.bayrakli
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from Fonksiyonlar import weighted_edit_distance

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
        
        # ✨ YENİ: Cause code ve şebeke unsuru mappings
        self.cause_code_to_unsur = structures.cause_code_to_unsur
        self.unsur_to_categories = structures.unsur_to_categories
        
        # Global yapılar (eski)
        self.category_phrase_ids = structures.category_phrase_ids
        self.category_keyword_ids = structures.category_keyword_ids
        self.category_all_ids = structures.category_all_ids
        
        # ✨ YENİ: Şebeke unsuru bazlı yapılar
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
        """Kelimenin kökünü bul"""
        try:
            analysis = self.zemb.analyze_sentence(self.tr_lower(word))
            if analysis and len(analysis) > 0:
                if "lemmas" in analysis[0] and analysis[0]["lemmas"]:
                    lemma = analysis[0]["lemmas"][0]
                    if lemma and lemma != "UNK":
                        return self.tr_lower(lemma)
                    
                    else:
                        # print(analysis[0]["token"])
                        pass
        except:
            pass
        return self.tr_lower(word)
    
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
            lemmas.append(lemma)
            
            # Lemma'yı ID'ye çevir
            if lemma in self.lemma2id:
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
        
        # === 3. YENİ: SINGLE TOKEN MATCHING ===
        # Her bir input ID'sinin hangi kategorilerde geçtiğini bul
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
                    
                    if matched_texts:
                        single_token_matches.append({
                            'kategori': kategori,
                            'type': 'single_token',
                            'match_type': 'single_token',
                            'token': self.id2lemma[input_id],
                            'matched_id': input_id,
                            'matched_ids': [input_id],
                            'found_in': matched_texts,
                            'confidence': 0.6  # Tek kelime için daha düşük güven
                        })
        
        # Duplicate'leri temizle (aynı kategori + aynı token)
        single_token_matches = self._deduplicate_single_tokens(single_token_matches)
        
        return exact_matches, subset_matches, partial_matches, single_token_matches

    def _deduplicate_single_tokens(self, single_token_matches):
        """Aynı kategori + aynı token için sadece bir entry tut"""
        seen = {}
        unique = []
        
        for match in single_token_matches:
            key = (match['kategori'], match['token'])
            if key not in seen:
                seen[key] = match
                unique.append(match)
            else:
                # Birden fazla phrase/keyword'de geçiyorsa birleştir
                seen[key]['found_in'].extend(match['found_in'])
        
        return unique
    
    def calculate_idf_weights(self):
        """
        Her ID için IDF (Inverse Document Frequency) ağırlığı hesapla
        Çok kategoride geçen ID'ler düşük ağırlık alır
        """
        if hasattr(self, 'idf_weights'):
            return self.idf_weights
        
        id_to_category_count = defaultdict(int)
        total_categories = len(self.category_all_ids)
        
        # Her ID kaç kategoride var?
        for cat_ids in self.category_all_ids.values():
            for id_val in cat_ids:
                id_to_category_count[id_val] += 1
        
        # IDF hesapla: log(toplam_kategori / id_geçtiği_kategori_sayısı)
        self.idf_weights = {}
        for id_val, count in id_to_category_count.items():
            self.idf_weights[id_val] = np.log(total_categories / count)
        
        return self.idf_weights
    
    def calculate_category_scores_with_idf(self, exact_matches, subset_matches, 
                                           partial_matches, single_token_matches):
        """
        IDF ağırlıkları ile kategori skorları hesapla
        """
        idf_weights = self.calculate_idf_weights()
        
        category_scores = defaultdict(lambda: {
            'total_score': 0.0,
            'matches': [],
            'match_types': defaultdict(int),
            'max_confidence': 0.0,
            'matched_ids': set(),
            'single_tokens': []
        })
        
        all_matches = exact_matches + subset_matches + partial_matches
        
        for match in all_matches:
            cat = match['kategori']
            conf = match['confidence']
            match_type = match['match_type']
            matched_ids = match['matched_ids']
            
            # IDF ağırlıklarını hesapla
            avg_idf = np.mean([idf_weights.get(id_val, 1.0) for id_val in matched_ids])
            
            # Skor hesapla (IDF ile çarp)
            if match['type'] == 'phrase':
                if match_type == 'exact':
                    score = 10.0 * avg_idf
                elif match_type == 'subset_sequential':
                    score = 7.0 * avg_idf
                elif match_type == 'subset':
                    score = 5.0 * avg_idf
                else:  # partial
                    score = 3.0 * match.get('coverage', 0.5) * avg_idf
            else:  # keyword
                base_score = 4.0 if match_type == 'exact' else 3.0
                score = base_score * avg_idf
            
            category_scores[cat]['total_score'] += score
            category_scores[cat]['matches'].append(match)
            category_scores[cat]['match_types'][match_type] += 1
            category_scores[cat]['max_confidence'] = max(
                category_scores[cat]['max_confidence'], conf
            )
            category_scores[cat]['matched_ids'].update(matched_ids)
        
        # Single token matches
        for match in single_token_matches:
            cat = match['kategori']
            matched_id = match['matched_id']
            
            # IDF ağırlığı
            idf = idf_weights.get(matched_id, 1.0)
            score = 2.0 * idf
            
            category_scores[cat]['total_score'] += score
            category_scores[cat]['single_tokens'].append(match)
            category_scores[cat]['match_types']['single_token'] += 1
            category_scores[cat]['matched_ids'].add(matched_id)
        
        # Coverage (aynı)
        for cat in category_scores:
            total_ids_in_category = len(self.category_all_ids[cat])
            matched_ids_count = len(category_scores[cat]['matched_ids'])
            
            category_scores[cat]['coverage'] = matched_ids_count / total_ids_in_category
            
            if category_scores[cat]['coverage'] >= 0.3:
                has_strong_match = any(
                    m['type'] in ['phrase', 'keyword'] 
                    for m in category_scores[cat]['matches']
                )
                bonus_multiplier = 2.0 if has_strong_match else 1.0
                bonus = category_scores[cat]['coverage'] * bonus_multiplier
                category_scores[cat]['total_score'] += bonus
        
        return dict(category_scores)
    
    def _check_phrase_match(
        self, 
        input_ids: List[int], 
        input_id_set: Set[int], 
        phrase_info: Dict, 
        kategori: str
    ) -> Optional[Dict]:
        """
        Bir phrase'in input ile eşleşip eşleşmediğini kontrol et
        
        Args:
            input_ids: Input ID listesi
            input_id_set: Input ID seti
            phrase_info: Phrase bilgisi dict
            kategori: Kategori adı
        
        Returns:
            Match dict veya None
        """
        target_ids = phrase_info['ids']  # frozenset
        target_list = phrase_info['ids_list']  # list
        
        # A) Exact match (set equality)
        if target_ids == input_id_set:
            return {
                'kategori': kategori,
                'type': 'phrase',
                'match_type': 'exact',
                'text': phrase_info['phrase'],
                'matched_ids': list(target_ids),
                'confidence': 1.0
            }
        
        # B) Subset match (target tamamen input içinde)
        if target_ids.issubset(input_id_set):
            # Sıralı mı kontrol et (bonus için)
            is_sequential = self._check_sequential(input_ids, target_list)
            
            confidence = 0.9 if is_sequential else 0.85
            
            return {
                'kategori': kategori,
                'type': 'phrase',
                'match_type': 'subset_sequential' if is_sequential else 'subset',
                'text': phrase_info['phrase'],
                'matched_ids': list(target_ids),
                'confidence': confidence,
                'coverage': 1.0  # Phrase'in tamamı var
            }
        
        # C) Partial match (kesişim var ama tam değil)
        intersection = target_ids.intersection(input_id_set)
        if intersection:
            coverage = len(intersection) / len(target_ids)
            
            if coverage >= 0.5:  # En az %50 eşleşmeli
                confidence = coverage * 0.7  # Penalty
                
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
    
    
    def _check_keyword_match(
        self,
        input_id_set: Set[int],
        keyword_info: Dict,
        kategori: str
    ) -> Optional[Dict]:
        """
        Bir keyword'ün input ile eşleşip eşleşmediğini kontrol et
        
        Args:
            input_id_set: Input ID seti
            keyword_info: Keyword bilgisi dict
            kategori: Kategori adı
        
        Returns:
            Match dict veya None
        """
        target_ids = keyword_info['ids']  # frozenset
        
        # Keyword exact veya subset
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
    
    
    def _check_sequential(self, input_ids: List[int], target_ids: List[int]) -> bool:
        """
        Target ID'leri input'ta sıralı mı? (araya başka ID'ler girebilir)
        
        Args:
            input_ids: Input ID listesi
            target_ids: Target ID listesi
        
        Returns:
            bool: Sıralı ise True
        """
        if not target_ids:
            return True
        
        target_idx = 0
        for input_id in input_ids:
            if input_id == target_ids[target_idx]:
                target_idx += 1
                if target_idx == len(target_ids):
                    return True
        
        return False
    
    
    def _find_single_tokens(
        self,
        input_ids: List[int],
        kategori: str
    ) -> List[Dict]:
        """
        Input'taki her ID için kategori içinde single token match bul
        
        Args:
            input_ids: Input ID listesi
            kategori: Kategori adı
        
        Returns:
            Single token match listesi
        """
        matches = []
        seen_ids = set()  # Duplicate önleme
        
        for input_id in input_ids:
            # Bu ID zaten işlendi mi?
            if input_id in seen_ids:
                continue
            
            # Bu ID bu kategoride var mı?
            if input_id in self.category_all_ids.get(kategori, set()):
                # Bu ID'nin hangi phrase/keyword'e ait olduğunu bul
                found_in = []
                
                # Phrase'lerde ara
                for phrase_info in self.category_phrase_ids.get(kategori, []):
                    if input_id in phrase_info['ids']:
                        found_in.append({
                            'text': phrase_info['phrase'],
                            'type': 'phrase'
                        })
                
                # Keyword'lerde ara
                for keyword_info in self.category_keyword_ids.get(kategori, []):
                    if input_id in keyword_info['ids']:
                        found_in.append({
                            'text': keyword_info['keyword'],
                            'type': 'keyword'
                        })
                
                if found_in:
                    matches.append({
                        'kategori': kategori,
                        'type': 'single_token',
                        'match_type': 'single_token',
                        'token': self.id2lemma.get(input_id, f'ID_{input_id}'),
                        'matched_id': input_id,
                        'confidence': 0.6,
                        'found_in': found_in
                    })
                    
                    seen_ids.add(input_id)
        
        return matches
    
    def _format_predictions(self, sorted_categories: List[Tuple]) -> List[Dict]:
        """
        Tahminleri formatla
        
        Args:
            sorted_categories: Sıralanmış (kategori, info) tuple'ları
        
        Returns:
            Formatlanmış tahmin listesi
        """
        predictions = []
        
        for cat, info in sorted_categories:
            pred = {
                'kategori': cat,
                'confidence': round(info['normalized_confidence'], 4),
                'confidence_pct': info['confidence_pct'],
                'raw_score': round(info['total_score'], 2),
                'coverage': round(info.get('coverage', 0), 2),
                'match_count': len(info.get('matches', [])) + len(info.get('single_tokens', [])),
                'match_types': dict(info.get('match_types', {})),
                'matches': []
            }
            
            # Phrase/Keyword matches
            for m in info.get('matches', []):
                pred['matches'].append({
                    'text': m['text'],
                    'type': m['type'],
                    'match_type': m['match_type'],
                    'confidence': round(m['confidence'], 2)
                })
            
            # Single token matches
            if 'single_tokens' in info and info['single_tokens']:
                pred['single_token_matches'] = [
                    {
                        'token': st['token'],
                        'confidence': 0.6,
                        'found_in_count': len(st.get('found_in', [])),
                        'examples': [f['text'] for f in st.get('found_in', [])][:3]
                    }
                    for st in info['single_tokens']
                ]
            
            predictions.append(pred)
        
        return predictions
    
    def normalize_scores(self, category_scores):
        """
        Power scaling: Büyük skorları daha baskın yapar
        """
        if not category_scores:
            return {}
        
        scores = np.array([info['total_score'] for info in category_scores.values()])
        
        # Min-Max normalization (0-1 arası getir)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            normalized = np.ones(len(scores)) / len(scores)
        else:
            # 0-1 arası normalize et
            normalized_01 = (scores - min_score) / (max_score - min_score)
            
            # POWER TRANSFORM: Büyük değerleri güçlendir
            # power càng büyük, fark càng açılır
            power = 2.0  # 2.0, 3.0, veya 4.0 deneyin
            powered = np.power(normalized_01, power)
            
            # Toplam 1 yap
            normalized = powered / powered.sum()
        
        for (cat, info), norm_score in zip(category_scores.items(), normalized):
            info['normalized_confidence'] = float(norm_score)
            info['confidence_pct'] = round(float(norm_score) * 100, 2)
        
        return category_scores
    
    def evaluate_all_beams(self, result_beams, cause_code, matcher, threshold=0.1, 
                           normalization='power', norm_param=2, verbose=False):
        """Tüm beam node'larını değerlendir ve sonuçları birleştir"""
        
        # Her kategori için sonuçları topla
        category_scores = defaultdict(list)  # {kategori: [(confidence, node_quality, coverage), ...]}
        category_match_details = defaultdict(list)  # Detayları sakla
        
        for beam_idx, node in enumerate(result_beams):
            # Node kalitesini hesapla (concept operasyonları daha iyi)
            node_quality = self.calculate_node_quality(node)
            
            # Bu node için tahmin yap
            result = matcher.predict(
                " ".join(node.out_tokens), 
                cause_code, 
                top_k=5,  # Daha fazla aday al
                min_confidence=0.01
            )
            
            print(f"\n{'='*80}")
            print(f"🔍 BEAM NODE {beam_idx + 1}/{len(result_beams)}")
            print(f"   Kalite: {node_quality:.2f}")
            print(f"   Tokens: {' '.join(node.out_tokens)}")
            print(f"   Operasyonlar: {node.operations}")
            
            # Her tahmin için kaydet
            if result['predictions']:
                for pred in result['predictions']:
                    category_scores[pred['kategori']].append({
                        'confidence': pred['confidence_pct'],
                        'raw_score': pred['raw_score'],
                        'node_quality': node_quality,
                        'coverage': pred['coverage'],
                        'beam_idx': beam_idx,
                        'match_types': pred['match_types'],
                        'matches': pred['matches']
                    })
        
        # Nihai skorları hesapla
        final_predictions = []
        
        for kategori, scores in category_scores.items():
            # Strateji 1: Ağırlıklı ortalama (node kalitesi ile)
            weighted_conf = sum(s['confidence'] * s['node_quality'] for s in scores) / sum(s['node_quality'] for s in scores)
            
            # Strateji 2: Maksimum güven (en iyi node'u al)
            max_conf = max(s['confidence'] for s in scores)
            
            # Strateji 3: Ortalama coverage
            avg_coverage = sum(s['coverage'] for s in scores) / len(scores)
            
            # Strateji 4: Kaç node'da görüldü
            appearance_count = len(scores)
            appearance_ratio = appearance_count / len(result_beams)
            
            # Nihai skor: Birden fazla faktörü birleştir
            final_score = (
                weighted_conf * 0.5 +          # Ağırlıklı güven %50
                max_conf * 0.2 +                # En yüksek güven %20
                (avg_coverage * 100) * 0.1 +    # Ortalama coverage %10
                (appearance_ratio * 100) * 0.2  # Görülme oranı %20
            )
            
            final_predictions.append({
                'kategori': kategori,
                'final_score': final_score,
                'weighted_confidence': weighted_conf,
                'max_confidence': max_conf,
                'avg_coverage': avg_coverage,
                'appearance_count': appearance_count,
                'appearance_ratio': appearance_ratio,
                'beam_details': scores
            })
        
        # Nihai skora göre sırala
        final_predictions.sort(key=lambda x: x['final_score'], reverse=True)
        
        if not final_predictions:
            return []
        
        # Skorları normalize et
        raw_scores = np.array([p['final_score'] for p in final_predictions])
        
        if normalization == 'power':
            normalized_scores = self.normalize_scores_with_emphasis(raw_scores, power=norm_param)
        elif normalization == 'softmax':
            normalized_scores = self.softmax_normalize(raw_scores, temperature=norm_param)
        else:
            # Basit min-max
            min_s, max_s = raw_scores.min(), raw_scores.max()
            normalized_scores = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else np.ones_like(raw_scores)
        
        # Normalize edilmiş skorları ekle
        for pred, norm_score in zip(final_predictions, normalized_scores):
            pred['normalized_score'] = float(norm_score)
            pred['normalized_score_pct'] = float(norm_score * 100)
        
        
        filtered = [
            pred for pred in final_predictions 
            if pred['normalized_score'] >= threshold
        ]
        
        if verbose:
            print(f"\n📊 Normalizasyon ({normalization}):")
            for pred in final_predictions[:5]:
                print(f"   {pred['kategori']}: "
                      f"Raw={pred['final_score']:.2f} → "
                      f"Norm={pred['normalized_score']:.4f} ({pred['normalized_score_pct']:.2f}%)")
        
        return filtered

    def normalize_scores_with_emphasis(self, scores, power=2):
        """Power scaling ile normalize et"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.ones_like(scores) / len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        emphasized = normalized ** power
        return emphasized / emphasized.sum()

        
    def calculate_node_quality(self, node):
        """Node kalitesini hesapla (0-1 arası)"""
        concept_ops = sum(1 for op in node.operations if 'CONCEPT' in op)
        fallback_ops = sum(1 for op in node.operations if 'FALLBACK' in op or 'UNKNOWN' in op)
        total_ops = len(node.operations)
        
        if total_ops == 0:
            return 0.5
        
        # Concept oranı yüksek = iyi
        concept_ratio = concept_ops / total_ops
        # Fallback oranı düşük = iyi
        fallback_penalty = fallback_ops / total_ops
        
        quality = concept_ratio * 0.7 + (1 - fallback_penalty) * 0.3
        return max(0.1, min(1.0, quality))  # 0.1-1.0 arası sınırla
    
    def predict(
        self, 
        user_input: str, 
        cause_code: str,
        top_k: int = 5, 
        min_confidence: float = 0.01
        ):
        """
        Ana tahmin fonksiyonu - İki aşamalı
        
        Args:
            user_input: Kullanıcı metni
            cause_code: Cause code
            top_k: En iyi kaç kategori
            min_confidence: Minimum güven skoru
        
        Returns:
            Tahmin sonuçları
        """
        # ✨ 1. Cause Code → Şebeke Unsuru
        sebeke_unsuru = self.cause_code_to_unsur.get(cause_code)
        
        # ✨ 2. Şebeke Unsuru kontrolü
        search_all_categories = False
        relevant_categories = set()
        
        if not sebeke_unsuru or sebeke_unsuru == "-" or sebeke_unsuru.strip() == "":
            # Şebeke unsuru yok veya "-" ise TÜM kategorilerde ara
            print(f"⚠️ Cause code '{cause_code}' için şebeke unsuru belirtilmemiş")
            print(f"   → Tüm kategorilerde arama yapılacak")
            
            search_all_categories = True
            relevant_categories = set(self.category_phrase_ids.keys())
            sebeke_unsuru = "TÜM ŞEBEKE UNSURLARI"
        else:
            # Belirli şebeke unsurunda ara
            relevant_categories = self.unsur_to_categories.get(sebeke_unsuru, set())
            
            if not relevant_categories:
                return {
                    'input': user_input,
                    'cause_code': cause_code,
                    'sebeke_unsuru': sebeke_unsuru,
                    'error': f'Şebeke unsuru için kategori bulunamadı: {sebeke_unsuru}',
                    'predictions': []
                }
        
        print(f"\n🔍 Arama Kapsamı:")
        print(f"   Cause Code: {cause_code}")
        print(f"   Şebeke Unsuru: {sebeke_unsuru}")
        print(f"   Arama Modu: {'GLOBAL (Tüm kategoriler)' if search_all_categories else 'FİLTRELİ'}")
        print(f"   İlgili Kategoriler ({len(relevant_categories)}): {list(relevant_categories)[:5]}...")
        
        # 3. Input'u ID'lere çevir
        input_ids, input_lemmas, input_words = self.text_to_lemma_ids(user_input)
        input_id_set = set(input_ids)
        
        if not input_ids:
            return {
                'input': user_input,
                'cause_code': cause_code,
                'sebeke_unsuru': sebeke_unsuru,
                'search_mode': 'global' if search_all_categories else 'filtered',
                'error': 'Geçerli kelime bulunamadı',
                'predictions': []
            }
        
        # 4. ✨ Arama yap (global veya filtered)
        if search_all_categories:
            # TÜM KATEGORİLERDE ARA (eski yöntem)
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
        
        # 5. Skorları hesapla
        category_scores = self.calculate_category_scores_with_idf(
            exact, subset, partial, single_token
        )
        
        # 6. Normalize et
        category_scores = self.normalize_scores(category_scores)
        
        # 7. Sırala ve filtrele
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1]['normalized_confidence'],
            reverse=True
        )
        
        sorted_categories = [
            (cat, info) for cat, info in sorted_categories
            if info['normalized_confidence'] >= min_confidence
        ][:top_k]
        
        # 8. Çıktı formatla
        predictions = self._format_predictions(sorted_categories)
        
        return {
            'input': user_input,
            'cause_code': cause_code,
            'sebeke_unsuru': sebeke_unsuru,
            'search_mode': 'global' if search_all_categories else 'filtered',
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
        ✨ YENİ: TÜM kategorilerde ara (şebeke unsuru yok ise)
        """
        exact_matches = []
        subset_matches = []
        partial_matches = []
        single_token_matches = []
        
        # TÜM kategorilerde ara (eski yöntem)
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
        
        # ✨ Sadece bu şebeke unsurundaki kategorilerde ara
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


# ============================================
# KULLANIM
# ============================================

# if __name__ == "__main__":
    # from zemberek import TurkishMorphology
    
    # # 1. Zemberek başlat
    # print("🔧 Zemberek yükleniyor...")
    # # zemb = TurkishMorphology.create_with_defaults()
    # zemb = ZemberekClient(host="localhost", port=6789)
    # dokunulmayacak_kelimeler = kelimeleri_al(find_file("dokunulmayacak_kelimeler.txt"))
    # # 2. Matcher oluştur
    # print("\n🔧 Matcher oluşturuluyor...")
    # matcher = IDBasedCategoryMatcher('arızalar_all.xlsx', zemb, dokunulmayacak_kelimeler)
    
    # self = IDBasedCategoryMatcher('arızalar_all.xlsx', zemb, dokunulmayacak_kelimeler)
    
    # # 3. Debug bilgisi
    # matcher.debug_info("Kırılma")
    
    # # 4. Test
    # test_cases = [
    #     "trafo büyük kol kırık var",
    #     "ayırıcı şiddetli kırıldı",
    #     "trafo izolasyon hatası",
    #     "bushing patlama ve kaçak",
    #     "trafo yandı ve yağ kaçak kirlendi ve kırık var"
    # ]
    
    # print("\n" + "="*80)
    # print("TAHMİN SONUÇLARI")
    # print("="*80)
    
    # for test_input in test_cases:
    #     result = matcher.predict(test_input, top_k=3, min_confidence=0.01)
        
    #     print(f"\n📝 INPUT: {result['input']}")
    #     print(f"🔤 IDs: {result['input_ids']}")
    #     print(f"🔤 Lemmas: {result['input_lemmas']}")
    #     print(f"🔍 Exact: {result['total_exact_matches']}, "
    #           f"Subset: {result['total_subset_matches']}, "
    #           f"Partial: {result['total_partial_matches']}")
        
    #     if result['predictions']:
    #         print(f"\n🎯 TAHMİNLER:")
    #         for i, pred in enumerate(result['predictions'], 1):
    #             print(f"\n  {i}. {pred['kategori']}")
    #             print(f"     Güven: {pred['confidence_pct']}% (raw: {pred['raw_score']})")
    #             print(f"     Coverage: {pred['coverage']*100:.0f}%")
    #             print(f"     Match types: {pred['match_types']}")
    #             print(f"     Matches:")
    #             for m in pred['matches'][:3]:  # İlk 3'ü göster
    #                 print(f"       - {m['text']} ({m['match_type']}, conf: {m['confidence']})")
    #     else:
    #         print("     ❌ Eşleşme yok")
        
    #     print("-" * 80)