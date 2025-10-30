# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:21:04 2025

@author: vural.bayrakli
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple


class LMScorerIface:
    def score_candidate(self, left: List[str], cand: str, right: List[str], original: str = "") -> float:
        raise NotImplementedError

class BertLMScorer(LMScorerIface):
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased", method="log_scale", concept_dict=None, device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        if device:
            self.model.to(device)
        self.device = device or "cpu"
        self.concept_dict = concept_dict or set()
        self.method = method

    def score_token(self, left: List[str], cand: str, right: List[str], original: str = "") -> float:
        """
        left: sol bağlam kelimeleri listesi
        cand: test edilen aday kelime
        right: sağ bağlam kelimeleri listesi
        return: float skor (daha yüksek = daha olası)
        """
        # Cümleyi [MASK] ile oluştur
        sentence = " ".join(left + ["[MASK]"] + right)
        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.device)

        # Mask token indeksini bul
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            mask_logits = outputs.logits[0, mask_token_index, :].squeeze()

        probs = F.softmax(mask_logits, dim=-1)

        # Aday kelimeyi tokenlere böl
        tokens = self.tokenizer.tokenize(cand)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if not token_ids:
            return 0.0

        # Çok token’lı kelimeler için ortalama skor
        if len(token_ids) > 1:
            score = sum(probs[tid].item() for tid in token_ids if tid < len(probs)) / len(token_ids)
        else:
            score = probs[token_ids[0]].item()

        # Domain sözlüğü boost (opsiyonel)
        # if cand in self.concept_dict:
        #     score *= 1.2       
            
        # return score
        return self.optimize_lm_score(score)
    
    def optimize_lm_score(self, raw_score: float) -> float:
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
        normalized = 1 - (log_score / 10.0)
        
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

class BertLMScorer_v_0_1(LMScorerIface):
    
    def __init__(self, model_name="dbmdz/bert-base-turkish-cased", concept_dict=None, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.concept_dict = concept_dict

    def pseudo_log_likelihood(self, tokens):
        """
        tokens: kelime listesi (örn. ['okula','gidiyorum'])
        return: toplam log-olasılık (daha yüksek = daha olası)
        """
        text = " ".join(tokens)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"][0]
        total_logprob = 0.0

        with torch.no_grad():
            for i in range(1, len(input_ids)-1):  # [CLS], [SEP] hariç
                masked = input_ids.clone()
                masked[i] = self.tokenizer.mask_token_id
                logits = self.model(input_ids=masked.unsqueeze(0)).logits
                log_probs = F.log_softmax(logits[0, i], dim=-1)
                total_logprob += log_probs[input_ids[i]].item()

        return total_logprob

    def score_token(self, left, cand, right):
        tokens = left + [cand] + right
        
        if cand in self.concept_dict:
            return self.pseudo_log_likelihood(tokens) + 10 
        else:
            return self.pseudo_log_likelihood(tokens)


# concept_dict = {"sigorta", "arıza", "trafo"}
# scorer = BertLMScorer(concept_dict=concept_dict)

# scorer2 = BertLMScorer_v_0_1(concept_dict=concept_dict)

# # tokens = ["sigorta", "içtim", "şeklinde", "arıza", "bildirildi"]

# def test_fonk(tokens):
#     total_score = 0.0

#     # 1 kelimelik (unigram) skorlar
#     for i, token in enumerate(tokens):
#         left = tokens[max(0, i-2):i]
#         right = tokens[i+1:i+3]
#         score = scorer2.score_token(left=left, cand=token, right=right)
#         total_score += score
#         print(f"Unigram '{token}' skoru:", score)
    
#     # 2 kelimelik (bigram) skorlar
#     for i in range(len(tokens) - 1):
#         bigram = " ".join(tokens[i:i+2])
#         left = tokens[max(0, i-2):i]
#         right = tokens[i+2:i+4]
        score = scorer2.score_token(left=[], cand="kaçak", right=[])
        score
#         total_score += score
#         print(f"Bigram '{bigram}' skoru:", score)
        
#     all_score = scorer2.score_token(
#         left=left,
#         cand=" ".join(tokens),
#         right=right
#     )
            
#     total_score += all_score
    
#     return total_score

# tokens = ['elektrik', 'ariza', 'nedeniyle', 'kesinti', 'var']
# tokens = ["ve", "arıza", "sayaç", "değişimi", "planlandı"]
# tokens = ["kacak", "akım", "tespit", "edildi"]

# test_fonk(tokens)


# print("Toplam skor:", total_score)

# print(cand_score)   





















