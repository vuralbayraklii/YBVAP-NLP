# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:38:26 2025

@author: vural.bayrakli
"""

import pandas as pd
from tqdm import tqdm  # Progress bar için
from Fonksiyonlar import tr_lower

def process_all_data(_input, zemb, corrector, STOP_WORDS, df, structures):
    """
    Tüm input verisini işleyip sonuçları DataFrame'e kaydeder
    """
    
    # Sonuçları saklamak için liste
    results_list = []
    
    # OG Fider olmayanları filtrele
    og_fider_olamayanlar = _input[_input["cause code"] != "OG Fider Açması"].copy()
    
    print(f"🔄 Toplam {len(og_fider_olamayanlar)} kayıt işlenecek...\n")
    
    # Her satırı işle
    for idx in tqdm(df.index, desc="İşleniyor"):
        try:
            # Veriyi al
            row = _input.loc[idx]
            tokens_raw = row["Concatted"]
            cause_code = row["cause code"]
            
            # Tokenize
            tokenized = zemb.tokenize(tokens_raw)
            
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
            
            # Spell correction
            result_beam = corrector.correct(tokens, verbose=False)
            
            # Stop words filtrele
            filtered_result = [t for t in result_beam[0].out_tokens if t not in STOP_WORDS]
            
            # Matcher ile tahmin
            matcher = IDBasedCategoryMatcher(df, zemb, structures)
            predictions = matcher.evaluate_all_beams(
                result_beam, 
                cause_code, 
                matcher, 
                normalization='power', 
                norm_param=2,
                verbose=False  # Tek tek yazdırma
            )
            
            # En iyi tahmini bul
            result_best = predictions[0]['kategori'] if predictions else "NO_MATCH"
            
            # Full predictions text oluştur
            if predictions:
                result_full_parts = []
                for i, pred in enumerate(predictions[:5], 1):  # İlk 5 tahmini al
                    pred_text = (
                        f"{i}. {pred['kategori']} "
                        f"(Güven: {pred['normalized_score_pct']:.2f}%, "
                        f"Weighted: {pred['weighted_confidence']:.2f}%, "
                        f"Coverage: {pred['avg_coverage']*100:.0f}%, "
                        f"Görülme: {pred['appearance_count']}/{len(result_beam)})"
                    )
                    result_full_parts.append(pred_text)
                result_full = " | ".join(result_full_parts)
            else:
                result_full = "NO_MATCH"
            
            # Sonucu listeye ekle
            results_list.append({
                'input_concatted': tokens_raw,
                'cause_code': cause_code,
                'result_best': result_best,
                'result_full': result_full,
                # 'corrected_tokens': " ".join(filtered_result),  # Bonus: düzeltilmiş tokenlar
                'num_predictions': len(predictions) if predictions else 0
            })
            
        except Exception as e:
            # Hata durumunda da kaydet
            results_list.append({
                'input_concatted': row["Concatted"],
                'cause_code': row["cause code"],
                'result_best': f"ERROR: {str(e)}",
                'result_full': f"ERROR: {str(e)}",
                'corrected_tokens': "",
                'num_predictions': 0
            })
            print(f"\n⚠️ Hata (index {idx}): {str(e)}")
    
    # DataFrame oluştur
    results_df = pd.DataFrame(results_list)
    
    print(f"\n✅ İşlem tamamlandı! Toplam {len(results_df)} kayıt.")
    print(f"📊 NO_MATCH sayısı: {(results_df['result_best'] == 'NO_MATCH').sum()}")
    print(f"📊 Başarılı eşleşme: {(results_df['result_best'] != 'NO_MATCH').sum()}")
    
    return results_df


# # KULLANIM:
# # ========
# final_df = process_all_data(
#     _input=_input.iloc[:25],
#     zemb=zemb,
#     corrector=corrector,
#     STOP_WORDS=STOP_WORDS,
#     df=df,
#     structures=structures
# )

# # CSV olarak kaydet
# final_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
# print("\n💾 Sonuçlar 'prediction_results.csv' dosyasına kaydedildi.")

# # Excel olarak kaydet (opsiyonel)
# final_df.to_excel('prediction_results.xlsx', index=False, engine='openpyxl')
# print("💾 Sonuçlar 'prediction_results.xlsx' dosyasına kaydedildi.")

# # İlk 10 satırı göster
# print("\n📋 İlk 10 kayıt:")
# print(final_df.head(10))

# # Özet istatistikler
# print("\n📈 ÖZET İSTATİSTİKLER:")
# print(f"Toplam kayıt: {len(final_df)}")
# print(f"Unique cause codes: {final_df['cause_code'].nunique()}")
# print(f"Unique best predictions: {final_df['result_best'].nunique()}")
# print(f"\nEn çok tahmin edilen kategoriler:")
# print(final_df['result_best'].value_counts().head(10))