import os
from ZemberekClient import ZemberekClient
from Fonksiyonlar import *
import pandas as pd
from idBasedCategoryMatcher import IDBasedCategoryMatcher
from config import NLPConfig, PathType
from arıza_işleme import *
import random
from tqdm import tqdm
from datetime import datetime
from process import process_all_data
from NLPv0_3 import ZemberekClientIface, BeamSearchCorrector, Weights, Thresholds, STOP_WORDS
import json

# -----------------------------
# Değişkenler
# -----------------------------

config_path = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\config\config.json"

config = NLPConfig.from_json(config_path, "vural.bayrakli")

# config dosyası oku
with open(config_path, 'r', encoding='utf-8') as f:
    config_data = json.load(f)

output_dir = r"C:\Users\vural.bayrakli\OneDrive - MRC\İletişim sitesi - MRC2024-104-ADM-GDZ - Yerli Buyuk Veri Analitigi Platformu\05_Proje Çalışmaları\Vural\TextMiningwithGit\YBVAP-NLP\output"
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    
    # Zemberek client'ı başlat
    print("Zemberek client başlatılıyor...")
    
    zemb = ZemberekClient(host=config.host, port=config.port)
    zemb_cli = ZemberekClientIface(zemb)
    
    dokunma = pd.read_excel(config.get_path('dokunulmayacak_kelimeler_path'))["kelimeler"].values

    PATH_GÜNCELLEME = config_data["NLP"]["Path_Güncelleme"]
    # Arızalar verisini güncelle
    config.add_path_template(
        'arızalar_path',
    '{base}/{ana}/' + PATH_GÜNCELLEME["arızalar"]["path"],
        path_type=PathType.FILE
    )

    df = pd.read_excel(config.get_path('arızalar_path'), sheet_name=PATH_GÜNCELLEME["arızalar"]["sheet_name"])

    # Çözüm Açıklama Şebeke Unsuru verisi yolu güncellemesi
    config.add_path_template(
        'çözüm_açıklama_info_path',
        '{base}/{ana}/' + PATH_GÜNCELLEME["çözüm_açıklama_info"]["path"],
        path_type=PathType.FILE
    )

    çözüm_açıklama_info = pd.read_excel(config.get_path('çözüm_açıklama_info_path'), sheet_name= PATH_GÜNCELLEME["çözüm_açıklama_info"]["sheet_name"])
    cause_code_info = pd.read_excel(config.get_path('çözüm_açıklama_info_path'), sheet_name= PATH_GÜNCELLEME["cause_code_info"]["sheet_name"])

    structures = build_morphosemantic_structures_v3(
        df,
        zemb,
        dokunma,
        çözüm_açıklama_info,
        cause_code_info
    )

    _input = pd.read_excel(config.get_path('input_concatted_path'))
    
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
    
    matcher = IDBasedCategoryMatcher(df, zemb, structures)

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

