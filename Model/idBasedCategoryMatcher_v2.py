from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd
import numpy as np
from arıza_işleme import process_text_with_context_smart


class IDBasedCategoryMatcher:
    def __init__(self, df, zemb, structures):
        self.zemb = zemb
        self.df = df
        
        # Structures'ı yükle
        self._initialize_from_structures(structures)
    
    def _initialize_from_structures(self, structures):
        """Structures'dan veri yapılarını yükle"""
        # ✅ YENİ: Morfosemantik mappings
        self.token_to_id = structures.token_to_id  # "kapamak_Verb_NEG" → ID
        self.id_to_token = structures.id_to_token  # ID → MorphosemanticToken
        
        # Geriye uyumluluk için
        self.lemma2id = structures.lemma2id
        self.id2lemma = structures.id2lemma
        
        # Cause code ve şebeke unsuru mappings
        self.cause_code_to_unsur = structures.cause_code_to_unsur
        self.unsur_to_categories = structures.unsur_to_categories
        
        # Global yapılar
        self.category_phrase_ids = structures.category_phrase_ids
        self.category_keyword_ids = structures.category_keyword_ids
        self.category_all_ids = structures.category_all_ids
        
        # Şebeke unsuru bazlı yapılar
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
    
    def text_to_morphosemantic_ids(self, text: str) -> Tuple[List[int], List, List[str]]:
        """
        Metni morfosemantik ID'lere çevir (BAĞLAMLI ANALİZ)
        
        Args:
            text: Input metni
        
        Returns:
            (id_list, morphosemantic_tokens, original_words)
        """
        text = self.tr_lower(text)
        
        # ✅ Tüm metni birlikte analiz et (bağlam korunur)
        morphosemantic_tokens = process_text_with_context_smart(text, self.zemb)
        
        ids = []
        original_words = text.split()
        
        for token in morphosemantic_tokens:
            # Token'ın unique key'ini oluştur
            token_key = token.to_key()
            
            # Bu key structures'da var mı?
            if token_key in self.token_to_id:
                token_id = self.token_to_id[token_key]
                ids.append(token_id)
            else:
                # Fallback: Sadece lemma'ya bak (POS ve NEG bilgisi olmadan)
                if token.lemma in self.lemma2id:
                    ids.append(self.lemma2id[token.lemma])
        
        return ids, morphosemantic_tokens, original_words
    
    def find_matches(self, input_ids: List[int], input_id_set: Set[int]):
        """
        Input ID'leriyle eşleşen phrase, keyword ve tek kelimeleri bul
        
        Args:
            input_ids: Input ID listesi (sıralı)
            input_id_set: Input ID seti (hızlı arama için)
        
        Returns:
            (exact_matches, subset_matches, partial_matches, single_token_matches)
        """
        exact_matches = []
        subset_matches = []
        partial_matches = []
        single_token_matches = []
        
        # === 1. PHRASE MATCHING ===
        for kategori, phrase_list in self.category_phrase_ids.items():
            for phrase_info in phrase_list:
                match = self._check_phrase_match(
                    input_ids, 
                    input_id_set, 
                    phrase_info, 
                    kategori
                )
                
                if match:
                    if match['match_type'] == 'exact':
                        exact_matches.append(match)
                    elif 'subset' in match['match_type']:
                        subset_matches.append(match)
                    elif match['match_type'] == 'partial':
                        partial_matches.append(match)
        
        # === 2. KEYWORD MATCHING ===
        for kategori, keyword_list in self.category_keyword_ids.items():
            for keyword_info in keyword_list:
                match = self._check_keyword_match(
                    input_id_set,
                    keyword_info,
                    kategori
                )
                
                if match:
                    subset_matches.append(match)
        
        # === 3. SINGLE TOKEN MATCHING ===
        for kategori in self.category_all_ids.keys():
            single_matches = self._find_single_tokens(input_ids, kategori)
            single_token_matches.extend(single_matches)
        
        # Duplicate'leri temizle
        single_token_matches = self._deduplicate_single_tokens(single_token_matches)
        
        return exact_matches, subset_matches, partial_matches, single_token_matches
    
    def _check_phrase_match(
        self, 
        input_ids: List[int], 
        input_id_set: Set[int], 
        phrase_info: Dict, 
        kategori: str
    ) -> Optional[Dict]:
        """Bir phrase'in input ile eşleşip eşleşmediğini kontrol et"""
        target_ids = phrase_info['ids']  # frozenset
        target_list = phrase_info['ids_list']  # list
        
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
    
    def _check_keyword_match(
        self,
        input_id_set: Set[int],
        keyword_info: Dict,
        kategori: str
    ) -> Optional[Dict]:
        """Bir keyword'ün input ile eşleşip eşleşmediğini kontrol et"""
        target_ids = keyword_info['ids']  # frozenset
        
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
        """Target ID'leri input'ta sıralı mı?"""
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
        """Input'taki her ID için kategori içinde single token match bul"""
        matches = []
        seen_ids = set()
        
        for input_id in input_ids:
            if input_id in seen_ids:
                continue
            
            if input_id in self.category_all_ids.get(kategori, set()):
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
                    # ✅ Morfosemantik token bilgisini al
                    morpho_token = self.id_to_token.get(input_id)
                    display_token = morpho_token.to_display_lemma() if morpho_token else self.id2lemma.get(input_id, f'ID_{input_id}')
                    
                    matches.append({
                        'kategori': kategori,
                        'type': 'single_token',
                        'match_type': 'single_token',
                        'token': display_token,
                        'matched_id': input_id,
                        'confidence': 0.6,
                        'found_in': found_in
                    })
                    
                    seen_ids.add(input_id)
        
        return matches
    
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
                seen[key]['found_in'].extend(match['found_in'])
        
        return unique
    
    def calculate_idf_weights(self):
        """Her ID için IDF ağırlığı hesapla"""
        if hasattr(self, 'idf_weights'):
            return self.idf_weights
        
        id_to_category_count = defaultdict(int)
        total_categories = len(self.category_all_ids)
        
        for cat_ids in self.category_all_ids.values():
            for id_val in cat_ids:
                id_to_category_count[id_val] += 1
        
        self.idf_weights = {}
        for id_val, count in id_to_category_count.items():
            self.idf_weights[id_val] = np.log(total_categories / count)
        
        return self.idf_weights
    
    def calculate_category_scores_with_idf(self, exact_matches, subset_matches, 
                                           partial_matches, single_token_matches):
        """IDF ağırlıkları ile kategori skorları hesapla"""
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
            
            avg_idf = np.mean([idf_weights.get(id_val, 1.0) for id_val in matched_ids])
            
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
            
            idf = idf_weights.get(matched_id, 1.0)
            score = 2.0 * idf
            
            category_scores[cat]['total_score'] += score
            category_scores[cat]['single_tokens'].append(match)
            category_scores[cat]['match_types']['single_token'] += 1
            category_scores[cat]['matched_ids'].add(matched_id)
        
        # Coverage
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
    
    def softmax_normalize(self, scores, temperature=5.0):
        """Softmax ile normalize et"""
        scores = np.array(scores)
        scaled = scores / temperature
        exp_scores = np.exp(scaled - scaled.max())
        return exp_scores / exp_scores.sum()
    
    def predict(self, input_text: str, cause_code: str = None, 
                top_k: int = 5, min_confidence: float = 0.01) -> Dict:
        """
        Input text için kategori tahmini yap (MORFOSEMANTIK)
        
        Args:
            input_text: Input metni
            cause_code: Cause code (opsiyonel, filtreleme için)
            top_k: Kaç tahmin döndürülecek
            min_confidence: Minimum güven eşiği
        
        Returns:
            Tahmin sonuçları dict
        """
        # ✅ MORFOSEMANTIK ANALİZ (bağlam korunur)
        input_ids, morpho_tokens, original_words = self.text_to_morphosemantic_ids(input_text)
        input_id_set = set(input_ids)
        
        # Şebeke unsuru filtreleme (varsa)
        target_categories = None
        if cause_code and cause_code in self.cause_code_to_unsur:
            unsur = self.cause_code_to_unsur[cause_code]
            if unsur != "-":  # Global değilse
                target_categories = self.unsur_to_categories.get(unsur, set())
        
        # Eşleşmeleri bul
        exact_matches, subset_matches, partial_matches, single_token_matches = self.find_matches(
            input_ids, 
            input_id_set
        )
        
        # Filtreleme (eğer cause code varsa)
        if target_categories:
            exact_matches = [m for m in exact_matches if m['kategori'] in target_categories]
            subset_matches = [m for m in subset_matches if m['kategori'] in target_categories]
            partial_matches = [m for m in partial_matches if m['kategori'] in target_categories]
            single_token_matches = [m for m in single_token_matches if m['kategori'] in target_categories]
        
        # Skorları hesapla
        category_scores = self.calculate_category_scores_with_idf(
            exact_matches,
            subset_matches,
            partial_matches,
            single_token_matches
        )
        
        if not category_scores:
            return {
                'input': input_text,
                'input_ids': input_ids,
                'input_lemmas': [t.to_display_lemma() for t in morpho_tokens],
                'morphosemantic_info': [
                    {
                        'surface': t.surface_form,
                        'lemma': t.lemma,
                        'display_lemma': t.to_display_lemma(),
                        'pos': t.pos,
                        'is_negative': t.is_negative,
                        'morphemes': t.morphemes
                    }
                    for t in morpho_tokens
                ],
                'predictions': [],
                'total_exact_matches': 0,
                'total_subset_matches': 0,
                'total_partial_matches': 0,
                'total_single_token_matches': 0
            }
        
        # Skorları normalize et
        category_scores = self.normalize_scores(category_scores)
        
        # Sırala ve filtrele
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        sorted_categories = [
            (cat, info) for cat, info in sorted_categories
            if info.get('normalized_confidence', 0) >= min_confidence
        ][:top_k]
        
        # Formatla
        predictions = self._format_predictions(sorted_categories)
        
        return {
            'input': input_text,
            'input_ids': input_ids,
            'input_lemmas': [t.to_display_lemma() for t in morpho_tokens],
            'morphosemantic_info': [
                {
                    'surface': t.surface_form,
                    'lemma': t.lemma,
                    'display_lemma': t.to_display_lemma(),
                    'pos': t.pos,
                    'is_negative': t.is_negative,
                    'morphemes': t.morphemes
                }
                for t in morpho_tokens
            ],
            'predictions': predictions,
            'total_exact_matches': len(exact_matches),
            'total_subset_matches': len(subset_matches),
            'total_partial_matches': len(partial_matches),
            'total_single_token_matches': len(single_token_matches)
        }
    
    def _format_predictions(self, sorted_categories: List[Tuple]) -> List[Dict]:
        """Tahminleri formatla"""
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
        """Skorları normalize et (power scaling)"""
        if not category_scores:
            return {}
        
        scores = np.array([info['total_score'] for info in category_scores.values()])
        
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            normalized = np.ones(len(scores)) / len(scores)
        else:
            normalized_01 = (scores - min_score) / (max_score - min_score)
            power = 2.0
            powered = np.power(normalized_01, power)
            normalized = powered / powered.sum()
        
        for (cat, info), norm_score in zip(category_scores.items(), normalized):
            info['normalized_confidence'] = float(norm_score)
            info['confidence_pct'] = round(float(norm_score) * 100, 2)
        
        return category_scores
    
    def calculate_node_quality(self, node):
        """Node kalitesini hesapla (beam search için)"""
        concept_ops = sum(1 for op in node.operations if 'CONCEPT' in op)
        fallback_ops = sum(1 for op in node.operations if 'FALLBACK' in op or 'UNKNOWN' in op)
        total_ops = len(node.operations)
        
        if total_ops == 0:
            return 0.5
        
        concept_ratio = concept_ops / total_ops
        fallback_penalty = fallback_ops / total_ops
        
        quality = concept_ratio * 0.7 + (1 - fallback_penalty) * 0.3
        return max(0.1, min(1.0, quality))
    
    def evaluate_all_beams(self, result_beams, cause_code, 
                           normalization='power', norm_param=2, verbose=False):
        """Tüm beam node'larını değerlendir ve sonuçları birleştir"""
        
        category_scores = defaultdict(list)
        
        for beam_idx, node in enumerate(result_beams):
            node_quality = self.calculate_node_quality(node)
            
            # ✅ Morfosemantik analizli predict
            result = self.predict(
                " ".join(node.out_tokens), 
                cause_code, 
                top_k=5,
                min_confidence=0.01
            )
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"🔍 BEAM NODE {beam_idx + 1}/{len(result_beams)}")
                print(f"   Kalite: {node_quality:.2f}")
                print(f"   Tokens: {' '.join(node.out_tokens)}")
                print(f"   Morfosemantik:")
                for info in result.get('morphosemantic_info', []):
                    print(f"     - {info['surface']}: {info['display_lemma']} ({info['pos']}, NEG={info['is_negative']})")
            
            if result['predictions']:
                for pred in result['predictions']:
                    category_scores[pred['kategori']].append({
                        'confidence': pred['confidence_pct'],
                        'raw_score': pred['raw_score'],
                        'node_quality': node_quality,
                        'coverage': pred['coverage'],
                        'beam_idx': beam_idx,
                        'match_types': pred['match_types'],
                        'matches': pred.get('matches', []),
                        'single_token_matches': pred.get('single_token_matches', [])
                    })
        
        # Nihai skorları hesapla
        final_predictions = []
        
        for kategori, scores in category_scores.items():
            weighted_conf = sum(s['confidence'] * s['node_quality'] for s in scores) / sum(s['node_quality'] for s in scores)
            max_conf = max(s['confidence'] for s in scores)
            avg_coverage = sum(s['coverage'] for s in scores) / len(scores)
            appearance_count = len(scores)
            appearance_ratio = appearance_count / len(result_beams)
            
            final_score = (
                weighted_conf * 0.5 +
                max_conf * 0.2 +
                (avg_coverage * 100) * 0.1 +
                (appearance_ratio * 100) * 0.2
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
        
        final_predictions.sort(key=lambda x: x['final_score'], reverse=True)
        
        if not final_predictions:
            return []
        
        # Normalize
        raw_scores = np.array([p['final_score'] for p in final_predictions])
        
        if normalization == 'power':
            normalized_scores = self.normalize_scores_with_emphasis(raw_scores, power=norm_param)
        elif normalization == 'softmax':
            normalized_scores = self.softmax_normalize(raw_scores, temperature=norm_param)
        else:
            min_s, max_s = raw_scores.min(), raw_scores.max()
            normalized_scores = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else np.ones_like(raw_scores)
        
        for pred, norm_score in zip(final_predictions, normalized_scores):
            pred['normalized_score'] = float(norm_score)
            pred['normalized_score_pct'] = float(norm_score * 100)
        
        # Threshold filtrele (en yüksek skorun %95'i)
        max_norm_score = final_predictions[0]['normalized_score']
        threshold = max_norm_score * 0.95
        
        filtered = [
            pred for pred in final_predictions 
            if pred['normalized_score'] >= threshold
        ]
        
        if verbose:
            print(f"\n📊 Filtreleme: {len(final_predictions)} → {len(filtered)} aday")
        
        return filtered