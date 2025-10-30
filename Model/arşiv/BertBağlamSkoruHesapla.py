# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 16:04:46 2025

@author: vural.bayrakli
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F
self = HedefliDuzeltici(None)
class HedefliDuzeltici:
    def __init__(self, concept_dict):
        self.model_name = "dbmdz/bert-base-turkish-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.concept_dict = concept_dict
        
    def aday_kelime_bul(self, hatali_kelime, max_distance=2):
        """Concept dict'ten benzer kelimeleri bul"""
        from Levenshtein import distance
        
        adaylar = []
        for kelime in self.concept_dict:
            if distance(hatali_kelime.lower(), kelime.lower()) <= max_distance:
                adaylar.append(kelime)
        
        return adaylar
    
    def baglamsal_skorla(self, cumle, hatali_kelime, aday_kelimeler):
        """Her aday için BERT skoru hesapla"""
        if not aday_kelimeler:
            return []
        
        masked_cumle = cumle.replace(hatali_kelime, "[MASK]")
        inputs = self.tokenizer(masked_cumle, return_tensors="pt")
        
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            mask_logits = outputs.logits[0, mask_token_index, :].squeeze()
        
        # Softmax uygula
        probs = F.softmax(mask_logits, dim=-1)
        
        skorlar = []
        for kelime in aday_kelimeler:
            # Kelimeyi subword token'lara böl
            tokens = self.tokenizer.tokenize(kelime)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            if token_ids:
                # Birden fazla token varsa, ortalama skor al
                if len(token_ids) > 1:
                    skor = sum(probs[tid].item() for tid in token_ids if tid < len(probs)) / len(token_ids)
                else:
                    skor = probs[token_ids[0]].item()
                
                skorlar.append((kelime, skor))
        
        return sorted(skorlar, key=lambda x: x[1], reverse=True)
    
    def duzelt(self, cumle, threshold=0.01):
        """Cümledeki kelimeleri düzelt"""
        kelimeler = cumle.split()
        duzeltilmis = []
        
        for i, kelime in enumerate(kelimeler):
            # Concept dict'te tam eşleşme var mı?
            if kelime.lower() in [c.lower() for c in self.concept_dict]:
                duzeltilmis.append(kelime)
                continue
            
            # Benzer kelimeler bul
            adaylar = self.aday_kelime_bul(kelime)
            
            if adaylar:
                # BERT ile skorla
                skorlu_adaylar = self.baglamsal_skorla(cumle, kelime, adaylar)
                
                if skorlu_adaylar and skorlu_adaylar[0][1] > threshold:
                    en_iyi = skorlu_adaylar[0][0]
                    print(f"'{kelime}' -> '{en_iyi}' (skor: {skorlu_adaylar[0][1]:.4f})")
                    duzeltilmis.append(en_iyi)
                else:
                    duzeltilmis.append(kelime)
            else:
                duzeltilmis.append(kelime)
        
        return " ".join(duzeltilmis)

# Kullanım
concept_dict = ["sigorta", "arıza", "elektrik", "motor", "sigara", "yangın"]
duzeltici = HedefliDuzeltici(concept_dict)

# Test
test_cumleleri = [
    "sigora attı",
    "araba sigortasi yaptırdım",
    "sigora içmek yasak"
]

for cumle in test_cumleleri:
    print(f"\nOrijinal: {cumle}")
    
    # Örnek: "sigora" için adayları skorla
    if "sigora" in cumle:
        adaylar = ["sigorta", "sigara"]
        skorlar = duzeltici.baglamsal_skorla(cumle, "sigora", adaylar)
        print("Aday skorları:")
        for kelime, skor in skorlar:
            print(f"  {kelime}: {skor:.4f}")