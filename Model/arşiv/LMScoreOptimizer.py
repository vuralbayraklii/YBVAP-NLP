# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 15:15:59 2025

@author: vural.bayrakli
"""

# -*- coding: utf-8 -*-
"""
LM Skor Optimizasyon Yöntemleri
"""

import math
import numpy as np
from typing import List, Tuple

class LMScoreOptimizer:
    """LM skorlarını normalize eden ve ölçeklendiren sınıf"""
    
    def __init__(self, method='log_scale'):
        """
        method: 'log_scale', 'min_max', 'z_score', 'rank_based', 'adaptive'
        """
        self.method = method
        self.score_history = []  # Adaptif normalizasyon için
        self.min_score = float('inf')
        self.max_score = float('-inf')
        
    def optimize_lm_score(self, raw_score: float, left: List[str], 
                          cand: str, right: List[str]) -> float:
        """Raw LM skorunu optimize et"""
        
        if self.method == 'log_scale':
            return self._log_scale_optimization(raw_score)
        elif self.method == 'min_max':
            return self._min_max_normalization(raw_score)
        elif self.method == 'z_score':
            return self._z_score_normalization(raw_score)
        elif self.method == 'rank_based':
            return self._rank_based_scoring(raw_score)
        elif self.method == 'adaptive':
            return self._adaptive_scaling(raw_score)
        else:
            return raw_score

    # YÖNTEm 1: Logaritmik Ölçeklendirme
    def _log_scale_optimization(self, score: float) -> float:
        """
        Küçük skorları logaritmik olarak büyüt
        0.0002 -> ~0.37
        0.00002 -> ~0.23
        """
        if score <= 0:
            return -10.0  # Penalty for impossible scores
        
        # Log dönüşümü + pozitif shift
        # -log10(score) yaklaşımı: küçük değerler büyük skorlar verir
        log_score = -math.log10(score + 1e-10)
        
        # 0-1 arasına normalize et (yaklaşık)
        # Tipik aralık [0, 10] olduğunu varsayarak
        normalized = 1.0 - (log_score / 10.0)
        
        # Negatif olmamasını garanti et
        return max(0.0, normalized)

    # YÖNTEM 2: Min-Max Normalizasyon (Dinamik)
    def _min_max_normalization(self, score: float) -> float:
        """
        Görülen skorları [0, 1] aralığına normalize et
        """
        # İlk değerleri sakla
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)
        
        if self.max_score == self.min_score:
            return 0.5
        
        # [0, 1] aralığına normalize et
        normalized = (score - self.min_score) / (self.max_score - self.min_score)
        
        # Farkı vurgulamak için power transformation
        # Küçük farkları büyütür
        return normalized ** 0.5  # Karekök alarak farkları artır

    # YÖNTEM 3: Z-Score Normalizasyon
    def _z_score_normalization(self, score: float) -> float:
        """
        Standart sapma bazlı normalizasyon
        """
        self.score_history.append(score)
        
        if len(self.score_history) < 2:
            return score * 1000  # İlk değerler için basit ölçekleme
        
        # Son N skor üzerinden istatistik
        recent_scores = self.score_history[-100:]  # Son 100 skor
        mean = np.mean(recent_scores)
        std = np.std(recent_scores)
        
        if std == 0:
            return 0.5
        
        # Z-score hesapla
        z_score = (score - mean) / std
        
        # Sigmoid ile [0, 1] aralığına sıkıştır
        return 1.0 / (1.0 + math.exp(-z_score))

    # YÖNTEM 4: Rank-Based Scoring
    def _rank_based_scoring(self, score: float) -> float:
        """
        Skorları sıralamaya göre değerlendir
        """
        self.score_history.append(score)
        
        if len(self.score_history) < 2:
            return score * 1000
        
        # Skorun yüzdelik dilimini bul
        sorted_scores = sorted(self.score_history)
        rank = sorted_scores.index(score) / len(sorted_scores)
        
        return rank

    # YÖNTEM 5: Adaptif Ölçeklendirme
    def _adaptive_scaling(self, score: float) -> float:
        """
        Dinamik olarak ölçek faktörünü ayarla
        """
        # Tipik skor aralığını tespit et
        if score > 0:
            magnitude = math.floor(math.log10(abs(score)))
            # -5 için 100000, -4 için 10000, -3 için 1000
            scale_factor = 10 ** (-magnitude + 2)
            return score * scale_factor
        return 0.0


class ImprovedTokenScorer:
    """Geliştirilmiş Token Scorer - LM optimizasyonlu"""
    
    def __init__(self, zemb, lm, concept_lex, weights):
        self.zemb = zemb
        self.lm = lm
        self.concept_lex = concept_lex
        self.w = weights
        
        # LM optimizer - en iyi yöntemi seç
        self.lm_optimizer = LMScoreOptimizer(method='log_scale')
        
        # Score component tracking (debug için)
        self.last_score_components = {}
        
    def lm_score(self, left: List[str], cand: str, right: List[str]) -> float:
        """Optimize edilmiş LM skoru"""
        
        # Raw skor al
        raw_score = self.lm.score_token(left, cand, right)
        
        # Optimize et
        optimized_score = self.lm_optimizer.optimize_lm_score(
            raw_score, left, cand, right
        )
        
        # Debug için sakla
        self.last_score_components['lm_raw'] = raw_score
        self.last_score_components['lm_optimized'] = optimized_score
        
        return optimized_score
    
    def score_candidate(self, left: List[str], cand: str, right: List[str], 
                        original: str, operation: str = 'REPLACE') -> float:
        """
        Geliştirilmiş scoring - component tracking ile
        """
        components = {}
        
        # LM Score (optimize edilmiş)
        lm_score = self.lm_score(left, cand, right)
        components['lm'] = self.w.alpha_lm * lm_score
        
        # Morph Score
        morph_score = 1.0 if self.zemb.analyze_valid(cand) else 0.0
        components['morph'] = self.w.beta_morph * morph_score
        
        # Domain Boost
        domain_score = self.domain_boost(cand, left, right)
        components['domain'] = domain_score
        
        # Edit Distance (normalized)
        edit_score = self.edit_cost(cand, original)
        components['edit'] = -self.w.delta_edit * edit_score
        
        # Diacritic
        diacritic_score = self.diacritic_cost(cand, original)
        components['diacritic'] = -self.w.eps_diacrit * diacritic_score
        
        # Keyboard
        keyboard_score = self.keyboard_cost(cand, original)
        components['keyboard'] = -self.w.zeta_keyboard * keyboard_score
        
        # Operation bonus
        if operation == 'SPLIT':
            components['operation'] = self.w.eta_split
        elif operation == 'MERGE':
            components['operation'] = 0.2
        else:
            components['operation'] = 0.0
        
        # Toplam skor
        total_score = sum(components.values())
        
        # Debug için sakla
        self.last_score_components = components
        self.last_score_components['total'] = total_score
        
        return total_score
    
    def get_score_breakdown(self) -> dict:
        """Son skorlama detaylarını döndür (debug için)"""
        return self.last_score_components
    
    def domain_boost(self, cand: str, left: List[str], right: List[str]) -> float:
        """Domain boost hesapla"""
        # Mevcut implementasyon...
        pass
    
    def edit_cost(self, cand: str, original: str) -> float:
        """Edit cost hesapla"""
        # Mevcut implementasyon...
        pass
    
    def diacritic_cost(self, cand: str, original: str) -> float:
        """Diacritic cost hesapla"""
        # Mevcut implementasyon...
        pass
    
    def keyboard_cost(self, cand: str, original: str) -> float:
        """Keyboard cost hesapla"""
        # Mevcut implementasyon...
        pass


# YÖNTEM 6: Relative Scoring (Karşılaştırmalı)
class RelativeLMScorer:
    """
    Mutlak skor yerine göreceli skor kullan
    """
    
    def score_candidates_relative(self, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Adayları birbirine göre skorla
        """
        if not candidates:
            return []
        
        # Skorları çıkar
        scores = [score for _, score in candidates]
        
        # Min-max bul
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # Hepsi aynı
            return [(cand, 0.5) for cand, _ in candidates]
        
        # Göreceli skorlar
        relative_candidates = []
        for cand, score in candidates:
            # [0, 1] aralığına normalize
            relative_score = (score - min_score) / (max_score - min_score)
            
            # Farkları vurgula (sigmoid veya power)
            emphasized_score = self._emphasize_differences(relative_score)
            
            relative_candidates.append((cand, emphasized_score))
        
        return relative_candidates
    
    def _emphasize_differences(self, score: float, method='power') -> float:
        """
        Küçük farkları büyüt
        """
        if method == 'power':
            # x^0.5 küçük farkları büyütür
            return score ** 0.5
        elif method == 'sigmoid':
            # Sigmoid stretch
            # score'u [-3, 3] aralığına map et sonra sigmoid
            stretched = (score - 0.5) * 6
            return 1.0 / (1.0 + math.exp(-stretched))
        elif method == 'log':
            # Log transformation
            if score <= 0:
                return 0
            if score >= 1:
                return 1
            # Log scale ile farkları artır
            return (math.log(score + 0.1) - math.log(0.1)) / (math.log(1.1) - math.log(0.1))
        else:
            return score


# ÖRNEK KULLANIM
class EnhancedBeamSearchCorrector:
    """
    LM optimizasyonlu Beam Search Corrector
    """
    
    def __init__(self, zemb, lm, concept_lex, weights=None, thresholds=None):
        self.zemb = zemb
        self.lm = lm
        
        # Geliştirilmiş scorer kullan
        self.scorer = ImprovedTokenScorer(zemb, lm, concept_lex, weights or Weights())
        self.th = thresholds or Thresholds()
        
        # Relative scorer for candidate comparison
        self.relative_scorer = RelativeLMScorer()
        
    def score_and_rank_candidates(self, candidates: List[str], 
                                 left: List[str], right: List[str], 
                                 original: str) -> List[Tuple[str, float]]:
        """
        Adayları skorla ve sırala - relative scoring ile
        """
        # Her aday için raw skor hesapla
        raw_scores = []
        for cand in candidates:
            score = self.scorer.score_candidate(left, cand, right, original)
            raw_scores.append((cand, score))
        
        # Relative scoring uygula
        relative_scores = self.relative_scorer.score_candidates_relative(raw_scores)
        
        # Sırala ve döndür
        relative_scores.sort(key=lambda x: x[1], reverse=True)
        
        return relative_scores
    
    def correct_with_debugging(self, tokens: List[str]) -> Tuple[List[str], dict]:
        """
        Düzeltme yap ve debug bilgisi döndür
        """
        result = self.correct(tokens)
        
        # Score breakdown'ları topla
        debug_info = {
            'input': tokens,
            'output': result,
            'score_components': self.scorer.get_score_breakdown()
        }
        
        return result, debug_info


# TEST
if __name__ == "__main__":
    # LM skor örnekleri
    test_scores = [0.0002, 0.00002, 0.000002, 0.001, 0.01]
    
    print("LM Skor Optimizasyon Testleri")
    print("="*50)
    
    for method in ['log_scale', 'min_max', 'adaptive']:
        optimizer = LMScoreOptimizer(method=method)
        print(f"\n{method.upper()} Yöntemi:")
        print("-"*30)
        
        for score in test_scores:
            optimized = optimizer.optimize_lm_score(score, [], "test", [])
            print(f"{score:12.8f} -> {optimized:8.4f}")
    
    print("\n" + "="*50)
    print("Göreceli Farklar:")
    print("-"*30)
    
    # İki skorun farkını göster
    optimizer = LMScoreOptimizer('log_scale')
    score1 = 0.0002
    score2 = 0.00002
    
    opt1 = optimizer.optimize_lm_score(score1, [], "test", [])
    opt2 = optimizer.optimize_lm_score(score2, [], "test", [])
    
    print(f"Orijinal fark: {score1/score2:.1f}x")
    print(f"Optimize fark: {opt1/opt2:.1f}x")
    print(f"Score1: {score1:.6f} -> {opt1:.4f}")
    print(f"Score2: {score2:.6f} -> {opt2:.4f}")