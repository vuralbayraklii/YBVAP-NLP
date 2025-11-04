def process_all_data(_input, zemb, matcher, corrector, STOP_WORDS, df, structures):
    """
    ✨ IDF SKORLAMALI - Tüm input verisini işleyip sonuçları DataFrame'e kaydeder
    
    Yeni Özellikler:
    - IDF skorları gösteriliyor
    - Raw confidence (IDF ağırlıklı) eklendi
    - Method dağılımı detaylı (direct_from_kök_neden, direct_from_kategori, text_matching_with_idf)
    - Şebeke unsuru ve arama modu bilgileri
    
    Args:
        _input: Input DataFrame
        zemb: Zemberek analyzer
        matcher: IDBasedCategoryMatcher instance (IDF skorlamalı)
        corrector: Spell corrector
        STOP_WORDS: Stop words (set, list veya dict olabilir)
        df: Arıza DataFrame
        structures: ArızaStructures instance
    
    Returns:
        results_df: Sonuçları içeren DataFrame
    """
    
    # ✨ FIX: STOP_WORDS'ü set'e çevir (dict ise)
    if isinstance(STOP_WORDS, dict):
        stop_words_set = set(STOP_WORDS.keys())
        print("⚠️  STOP_WORDS dict formatında, set'e çevrildi")
    elif isinstance(STOP_WORDS, list):
        stop_words_set = set(STOP_WORDS)
        print("⚠️  STOP_WORDS list formatında, set'e çevrildi")
    else:
        stop_words_set = STOP_WORDS  # Zaten set
    
    # Sonuçları saklamak için liste
    results_list = []
    
    print(f"\n{'='*80}")
    print(f"🔄 IDF SKORLAMALI SİSTEM - TOPLU İŞLEM BAŞLIYOR")
    print(f"{'='*80}")
    print(f"📊 Toplam {len(_input)} kayıt işlenecek...")
    print(f"✨ IDF skorları her tahmin için hesaplanacak\n")
    
    # Her satırı işle
    for idx in tqdm(_input.index, desc="İşleniyor", ncols=100):
        
        row = None  # Hata durumu için
        try:
            # Veriyi al
            
            row = _input.loc[idx]
            tokens_raw = row["Concatted"]
            cause_code = row["cause code"]
            çözüm_açıklama = row.get("Çözüm Açıklama", None)  # ✨ .get() ile güvenli erişim
            
            # # Tokenize
            # tokenized = zemb.tokenize(tokens_raw)
            
            # # Token işleme
            # final_tokens = []
            # for token, token_type in tokenized:
            #     if token_type in {'Word', 'WordWithSymbol', 'UnknownWord'}:
            #         if token_type in {'WordWithSymbol', 'UnknownWord'}:
            #             parts = token.replace('/', '-').split('-')
            #             final_tokens.extend([tr_lower(p) for p in parts if p])
            #         else:
            #             final_tokens.append(tr_lower(token))
            
            # tokens = final_tokens
            
            # # Spell correction - correct() beam list döndürür
            # result_beam = corrector.correct(tokens, verbose=False)
            
            # # İlk beam'den tokenları al ve stop words filtrele
            # if result_beam and len(result_beam) > 0:
            #     corrected_tokens = result_beam[0].out_tokens
            #     filtered_tokens = [t for t in corrected_tokens if t not in stop_words_set]
            # else:
            #     # Düzeltme başarısızsa orijinal tokenları kullan
            #     filtered_tokens = [t for t in tokens if t not in stop_words_set]
            
            # ✨ YENİ: IDF skorlamalı tahmin yap (print'leri bastır)
            import sys
            import io
            
            # Standart çıktıyı geçici olarak bastır
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                result = matcher.predict(
                    user_input=tokens_raw, 
                    cause_code=cause_code, 
                    çözüm_açıklama=çözüm_açıklama,
                    corrector=corrector,
                    top_k=5,
                    min_confidence=0.01
                )
            finally:
                # Standart çıktıyı geri yükle
                sys.stdout = old_stdout

            result
            
            # Predictions'ı al
            if isinstance(result, dict) and 'predictions' in result:
                predictions = result['predictions']
            else:
                predictions = result if isinstance(result, list) else []
            
            # En iyi tahmini bul
            result_best = predictions[0]['kategori'] if predictions else "NO_MATCH"
            
            # ✨ YENİ: IDF bilgilerini ekle
            best_confidence = predictions[0].get('confidence', 0.0) if predictions else 0.0
            best_raw_confidence = predictions[0].get('raw_confidence', 0.0) if predictions else 0.0

            result_best = ""
            result_full = ""

            if result.get('kök_neden', None) or result.get('kategori', None):
                kök_neden = result.get('kök_neden', None)
                kategori = result.get('kategori', None)
                sebeke_unsuru = result.get('şebeke_unsuru', None)
                processed_input = result.get('input', None)
                confidence = result["predictions"][0].get('confidence', 0.0)
          

            else:

                result_full_parts = []
                for i, pred in enumerate(predictions[:5], 1):
                    # Güven skorunu al
                    confidence = pred.get('confidence', 0.0)
                    raw_confidence = pred.get('raw_confidence', 0.0)
                    
                    # ✨ YENİ: IDF skorlu çıktı
                    pred_text = f"{i}. {pred['kategori']} (Conf: {confidence:.3f}"

                    sebeke_unsuru = pred.get('sebeke_unsuru', None)

                    processed_input = pred.get('input', None)
                    
                    # Raw confidence (IDF skorlu) ekle
                    if raw_confidence > 0:
                        pred_text += f", IDF: {raw_confidence:.3f}"
                    
                    # Match summary ekle
                    if 'match_summary' in pred:
                        summary = pred['match_summary']
                        if isinstance(summary, dict):
                            exact = summary.get('exact', 0)
                            subset = summary.get('subset', 0)
                            partial = summary.get('partial', 0)
                            single = summary.get('single_token', 0)
                            total = summary.get('total', 0)
                            
                            if total > 0:
                                match_str_parts = []
                                if exact > 0:
                                    match_str_parts.append(f"E:{exact}")
                                if subset > 0:
                                    match_str_parts.append(f"S:{subset}")
                                if partial > 0:
                                    match_str_parts.append(f"P:{partial}")
                                if single > 0:
                                    match_str_parts.append(f"ST:{single}")
                                
                                if match_str_parts:
                                    pred_text += f", [{'/'.join(match_str_parts)}]"
                    
                    pred_text += ")"
                    result_full_parts.append(pred_text)
                    
                result_full = " | ".join(result_full_parts)
            
            
            # ✨ YENİ: Ek bilgiler
            method = result.get('method', 'unknown') if isinstance(result, dict) else 'unknown'
            search_mode = result.get('search_mode', 'N/A') if isinstance(result, dict) else 'N/A'
            normalized_confidences = [result.get('normalized_confidences', 0.0) for pred in predictions] if predictions else []
            # Sonucu listeye ekle
            results_list.append({
                'input_concatted': tokens_raw,
                'cause_code': cause_code,
                'çözüm_açıklama': çözüm_açıklama if çözüm_açıklama else "",
                'şebeke_unsuru': sebeke_unsuru if sebeke_unsuru else "",
                'kök_neden': kök_neden if kök_neden else "",
                'kategori': kategori if kategori else "",
                'result_best': result_best,
                'result_full': result_full,
                'corrected_tokens': processed_input if processed_input else "",
                'num_predictions': len(predictions) if predictions else 0,
                'method': method,
                'search_mode': search_mode,  # ✨ YENİ
                'confidence': best_confidence,  # ✨ YENİ
                'normalized_confidences': normalized_confidences,  # ✨ YENİ
                'raw_confidence_idf': best_raw_confidence,  # ✨ YENİ (IDF skorlu)
            })
            
        except Exception as e:
            # Hata durumunda da kaydet
            error_msg = str(e)
            
            if row is not None:
                results_list.append({
                    'input_concatted': row.get("Concatted", "ERROR"),
                    'cause_code': row.get("cause code", "ERROR"),
                    'çözüm_açıklama': row.get("Çözüm Açıklama", ""),
                    'şebeke_unsuru': row.get("Şebeke Unsuru", ""),
                    'kök_neden': row.get("Kök Neden", ""),
                    'result_best': f"ERROR: {error_msg}",
                    'result_full': f"ERROR: {error_msg}",
                    'corrected_tokens': "",
                    'num_predictions': 0,
                    'method': 'error',
                    'search_mode': 'error',
                    'sebeke_unsuru': "",
                    'confidence': 0.0,
                    'raw_confidence_idf': 0.0,
                })
            else:
                results_list.append({
                    'input_concatted': f"ERROR at index {idx}",
                    'cause_code': "ERROR",
                    'çözüm_açıklama': "",
                    'result_best': f"ERROR: {error_msg}",
                    'result_full': f"ERROR: {error_msg}",
                    'corrected_tokens': "",
                    'num_predictions': 0,
                    'method': 'error',
                    'search_mode': 'error',
                    'sebeke_unsuru': "",
                    'confidence': 0.0,
                    'raw_confidence_idf': 0.0,
                })
            
            print(f"\n⚠️ Hata (index {idx}): {error_msg}")
            print(traceback.format_exc())
    
    # DataFrame oluştur
    results_df = pd.DataFrame(results_list)
    
    # ✨ YENİ: Detaylı İstatistikler
    print(f"\n{'='*80}")
    print("📊 IDF SKORLAMALI SİSTEM - İŞLEM SONUÇLARI")
    print(f"{'='*80}")
    
    # Temel istatistikler
    print(f"\n📈 GENEL İSTATİSTİKLER:")
    print(f"   ✅ Toplam işlenen kayıt: {len(results_df)}")
    
    no_match_count = (results_df['result_best'] == 'NO_MATCH').sum()
    success_count = ((results_df['result_best'] != 'NO_MATCH') & 
                     (~results_df['result_best'].str.startswith('ERROR'))).sum()
    error_count = results_df['result_best'].str.startswith('ERROR').sum()
    
    print(f"   ✅ Başarılı eşleşme: {success_count} (%{success_count/len(results_df)*100:.1f})")
    print(f"   ❌ NO_MATCH sayısı: {no_match_count} (%{no_match_count/len(results_df)*100:.1f})")
    
    if error_count > 0:
        print(f"   ⚠️  Hatalı kayıt: {error_count} (%{error_count/len(results_df)*100:.1f})")
    
    # Method dağılımı
    if 'method' in results_df.columns:
        print(f"\n📋 METHOD DAĞILIMI:")
        method_counts = results_df['method'].value_counts()
        for method, count in method_counts.items():
            pct = count / len(results_df) * 100
            print(f"   • {method}: {count} (%{pct:.1f})")
            
            # Method'a göre alt istatistikler
            if method in ['direct_from_kök_neden', 'direct_from_kategori']:
                print(f"     → Direkt mapping (IDF kullanılmadı)")
            elif method == 'text_matching_with_idf':
                print(f"     → IDF skorlamalı text matching")
    
    # Arama modu dağılımı
    if 'search_mode' in results_df.columns:
        print(f"\n🔍 ARAMA MODU DAĞILIMI:")
        search_mode_counts = results_df['search_mode'].value_counts()
        for mode, count in search_mode_counts.items():
            if mode not in ['error', 'N/A']:
                pct = count / len(results_df) * 100
                if mode == 'filtered':
                    print(f"   • 🎯 Filtered (Şebeke unsuru bazlı): {count} (%{pct:.1f})")
                elif mode == 'global':
                    print(f"   • 🌍 Global (Tüm kategoriler): {count} (%{pct:.1f})")
    
    # IDF skorları analizi (sadece text_matching_with_idf için)
    if 'raw_confidence_idf' in results_df.columns:
        idf_results = results_df[
            (results_df['method'] == 'text_matching_with_idf') & 
            (results_df['raw_confidence_idf'] > 0)
        ]
        
        if len(idf_results) > 0:
            print(f"\n💎 IDF SKOR ANALİZİ (Text Matching için):")
            print(f"   • Ortalama IDF Skor: {idf_results['raw_confidence_idf'].mean():.3f}")
            print(f"   • Medyan IDF Skor: {idf_results['raw_confidence_idf'].median():.3f}")
            print(f"   • Min IDF Skor: {idf_results['raw_confidence_idf'].min():.3f}")
            print(f"   • Max IDF Skor: {idf_results['raw_confidence_idf'].max():.3f}")
            
            # Yüksek IDF skorlu tahminler (>2.0)
            high_idf = idf_results[idf_results['raw_confidence_idf'] > 2.0]
            if len(high_idf) > 0:
                print(f"   ⭐ Yüksek IDF (>2.0): {len(high_idf)} (%{len(high_idf)/len(idf_results)*100:.1f})")
                print(f"      → Bu tahminler çok ayırt edici kelimeler içeriyor!")
    
    # Confidence dağılımı
    if 'confidence' in results_df.columns:
        conf_results = results_df[results_df['confidence'] > 0]
        
        if len(conf_results) > 0:
            print(f"\n📊 CONFIDENCE DAĞILIMI:")
            print(f"   • Ortalama Confidence: {conf_results['confidence'].mean():.3f}")
            print(f"   • Medyan Confidence: {conf_results['confidence'].median():.3f}")
            
            # Confidence aralıkları
            high_conf = conf_results[conf_results['confidence'] >= 0.8]
            med_conf = conf_results[(conf_results['confidence'] >= 0.5) & (conf_results['confidence'] < 0.8)]
            low_conf = conf_results[conf_results['confidence'] < 0.5]
            
            print(f"   • Yüksek Güven (≥0.8): {len(high_conf)} (%{len(high_conf)/len(conf_results)*100:.1f})")
            print(f"   • Orta Güven (0.5-0.8): {len(med_conf)} (%{len(med_conf)/len(conf_results)*100:.1f})")
            print(f"   • Düşük Güven (<0.5): {len(low_conf)} (%{len(low_conf)/len(conf_results)*100:.1f})")
    
    # Şebeke unsuru dağılımı
    if 'sebeke_unsuru' in results_df.columns:
        unsur_results = results_df[results_df['sebeke_unsuru'] != ""]
        
        if len(unsur_results) > 0:
            print(f"\n🏗️ ŞEBEKE UNSURU DAĞILIMI:")
            unsur_counts = unsur_results['sebeke_unsuru'].value_counts().head(10)
            for unsur, count in unsur_counts.items():
                pct = count / len(results_df) * 100
                print(f"   • {unsur}: {count} (%{pct:.1f})")
    
    print(f"\n{'='*80}")
    print("✅ İşlem tamamlandı!")
    print(f"{'='*80}\n")
    
    return results_df