import os
from ZemberekClient import ZemberekClient
from Fonksiyonlar import *
import pandas as pd
from idBasedCategoryMatcher import IDBasedCategoryMatcher
from config import NLPConfig
from arıza_işleme import *
import random
from tqdm import tqdm
from datetime import datetime
from process import process_all_data
from NLPv0_3 import ZemberekClientIface, BeamSearchCorrector, Weights, Thresholds, STOP_WORDS


# -----------------------------
# Değişkenler
# -----------------------------

config_path = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\config\config.json"

config = NLPConfig.from_json(config_path, "vural.bayrakli")

print(config.host)
print(config.port)
print(config.dokunulmayacak_kelimeler_path)
print(config.arızalar_path)
print(config.input_concatted_path)
print(config.çözüm_açıklama_to_cause_code_path)


output_dir = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\Vural\TextMiningwithGit\YBVAP-NLP\output"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    
    # Zemberek client'ı başlat
    print("Zemberek client başlatılıyor...")
    
    zemb = ZemberekClient(host=config.host, port=config.port)
    zemb_cli = ZemberekClientIface(zemb)
    
    dokunma = pd.read_excel(config.dokunulmayacak_kelimeler_path)["kelimeler"].values

    df = pd.read_excel(config.arızalar_path)
    
    çözüm_açıklama_to_cause_code = pd.read_excel(config.çözüm_açıklama_to_cause_code_path)    
    
    çözüm_açıklama_info = pd.read_excel(config.çözüm_açıklama_info_path, sheet_name="Çözüm_Açıklama_Şebeke_Unsuru")
    cause_code_info = pd.read_excel(config.çözüm_açıklama_info_path, sheet_name="cause_code_Şebeke_Unsuru")
    
    structures = build_morphosemantic_structures_v3(
        df,
        zemb,
        config.cause_code_şebeke_unsuru_path,
        dokunma,
        çözüm_açıklama_info,
        çözüm_açıklama_to_cause_code,
        cause_code_info
    )

    _input = pd.read_excel(config.input_concatted_path)
    
    sayilar = random.choices(range(1, len(_input)), k=250)
    sayilar = sorted(sayilar)
    
    input_sample = _input.iloc[sayilar]
    input_sample.reset_index(drop=True, inplace = True)
    
    # Corrector'ı oluştur
    corrector = BeamSearchCorrector(
        zemb_cli,  
        dokunma, 
        structures.lemma2id,
        weights=Weights(),
        thresholds=Thresholds(
            beam_width=10,
            tau_replace=0.30,
            tau_split=0.20,
            tau_merge=0.25,
            min_token_len_for_split=4
        )
    )
        
    self = BeamSearchCorrector(
        zemb_cli, 
        dokunma,
        structures.lemma2id,
        weights=Weights(),
        thresholds=Thresholds(
            beam_width=10,
            tau_replace=0.30,
            tau_split=0.20,
            tau_merge=0.25,
            min_token_len_for_split=4
        ))
    
    matcher = IDBasedCategoryMatcher(df, zemb, structures)
    self = matcher
        
    final_df = process_all_data(
        _input=_input,
        zemb=zemb,
        matcher=matcher,
        corrector=corrector,
        STOP_WORDS=STOP_WORDS,
        df=df,
        structures=structures
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(output_dir, f"prediction_results_{timestamp}.xlsx")

    # Excel olarak kaydet (opsiyonel)
    final_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"💾 Sonuçlar '{output_path}' dosyasına kaydedildi.")

