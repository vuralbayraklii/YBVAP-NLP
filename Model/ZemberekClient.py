# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:09:33 2025

@author: vural.bayrakli
"""

import unicodedata as ud
import re
import grpc

# === import your generated stubs (exactly like in your example) ===
import zemberek_grpc.language_id_pb2 as lang_pb
import zemberek_grpc.language_id_pb2_grpc as lang_svc
import zemberek_grpc.preprocess_pb2 as pre_pb
import zemberek_grpc.preprocess_pb2_grpc as pre_svc
import zemberek_grpc.normalization_pb2 as norm_pb
import zemberek_grpc.normalization_pb2_grpc as norm_svc
import zemberek_grpc.morphology_pb2 as morph_pb
import zemberek_grpc.morphology_pb2_grpc as morph_svc
import zemberek_grpc.spellcheck_pb2 as spellcheck
import zemberek_grpc.spellcheck_pb2_grpc as spellcheck_svc

# Türkçe duyarlı küçük harf
def tr_lower(s: str) -> str:
    return (s.replace("I", "ı").replace("İ", "i")).lower()

# Basit Levenshtein (küçük kelimeler için yeterli; O(len(a)*len(b)))
def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,          # sil
                dp[j - 1] + 1,      # ekle
                prev + (0 if a[i-1] == b[j-1] else 1)  # değiştir
            )
            prev = cur
    return dp[lb]

# Concept dict'i casefold edip hızlı arama yapalım
class ConceptLexicon:
    def __init__(self, words: set[str]):
        # hem orijinal hem lowercase map tut
        self.words_raw = set(words)
        self.words_lc  = {tr_lower(w) for w in words}

    def has(self, w: str) -> bool:
        return (w in self.words_raw) or (tr_lower(w) in self.words_lc)

    def nearest(self, w: str, max_dist: int = 2):
        """Kavram sözlüğünde w'ye en yakın adayı döndür (<= max_dist).
           Eşitlikte daha uzun eşleşmeyi tercih et (özgül terimler için)."""
        wl = tr_lower(w)
        best = None
        best_d = max_dist + 1
        for cand in self.words_lc:
            d = levenshtein(wl, cand)
            if d < best_d or (d == best_d and len(cand) > len(best or "")):
                best, best_d = cand, d
        if best is not None and best_d <= max_dist:
            # orijinal büyük/küçük harfini önemsemiyorsak sözlüğün "raw" halinden eşini bulalım
            for raw in self.words_raw:
                if tr_lower(raw) == best:
                    return raw
        return None

class ZemberekClient:
    def __init__(self, host="localhost", port=6789):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.lang = lang_svc.LanguageIdServiceStub(self.channel)
        self.prep = pre_svc.PreprocessingServiceStub(self.channel)
        self.norm = norm_svc.NormalizationServiceStub(self.channel)
        self.morph = morph_svc.MorphologyServiceStub(self.channel)
        # --- yeni eklenen SpellCheck servisi ---
        self.spell = spellcheck_svc.SpellCheckServiceStub(self.channel)

    # --- helpers ---
    @staticmethod
    def _basic_unicode_clean(text: str) -> str:
        t = ud.normalize("NFC", text)
        t = re.sub(r"[ \t\r\f\v]+", " ", t).strip()
        t = re.sub(r"([!?.,;:])\1+", r"\1", t)
        return t

    def detect_lang(self, text: str) -> str:
        try:
            return self.lang.Detect(lang_pb.LanguageIdRequest(input=text)).langId
        except Exception:
            return "unknown"

    def normalize_text(self, text: str) -> tuple[str, str]:
        try:
            resp = self.norm.Normalize(norm_pb.NormalizationRequest(input=text))
            norm_text = getattr(resp, "normalized_input", "") or getattr(resp, "normalizedInput", "")
            err = getattr(resp, "error", "")
            return (norm_text, "") if norm_text else (text, err)
        except Exception as e:
            return text, str(e)

    def tokenize(self, text: str):
        try:
            resp = self.prep.Tokenize(pre_pb.TokenizationRequest(input=text))
            return [(t.token, t.type) for t in resp.tokens]
        except Exception:
            return []

    def analyze_sentence(self, text: str):
        out = []
        lemmas = []
        try:
            resp = self.morph.AnalyzeSentence(morph_pb.SentenceAnalysisRequest(input=text))
            for r in resp.results:
                
                token = getattr(r, "token", "")
                best = getattr(r, "best", None)
                if best is not None and hasattr(best, "dictionaryItem") and hasattr(best.dictionaryItem, "lemma"):
                    lemmas.extend(best.lemmas)
                    # lemmas = list(getattr(best, "dictionaryItem", [])) or list(getattr(best, "lemma", []))
                    pos = getattr(best.dictionaryItem, "primaryPos", "") or getattr(best, "posShort", "")
                else:
                    lemmas, pos = [], ""
                out.append({"token": token, "lemmas": lemmas, "pos": pos})
        except Exception:
            pass
        return out
    
    def analyze_sentence_full(self, text: str):
        resp = None
        
        try:
            resp = self.morph.AnalyzeSentence(morph_pb.SentenceAnalysisRequest(input=text))
            
        except Exception:
            pass
        return resp

    # --- yeni SpellCheck metodları ---
    def spell_check(self, word: str) -> tuple[bool, str]:
        """Tek kelimenin doğru yazılıp yazılmadığını döner (is_correct, error)."""
        try:
                  
            word_capitalize = self._turkish_capitalize(word)
            
            resp = self.spell.Check(spellcheck.SpellCheckRequest(word=word))
            resp_capitalize = self.spell.Check(spellcheck.SpellCheckRequest(word=word_capitalize))
            
            resp_result = resp.is_correct or resp_capitalize.is_correct
            
            return resp_result, getattr(resp, "error", "")
        
        except Exception as e:
            return False, str(e)
        
    def _turkish_capitalize(self, word: str) -> str:
        """
        Türkçe karakterlere duyarlı capitalize
        
        Args:
            word: Capitalize edilecek kelime
        
        Returns:
            Capitalize edilmiş kelime
        """
        if not word:
            return word
        
        # Türkçe özel karakterler
        turkish_lower_to_upper = {
            'i': 'İ',
            'ı': 'I',
            'ğ': 'Ğ',
            'ü': 'Ü',
            'ş': 'Ş',
            'ö': 'Ö',
            'ç': 'Ç'
        }
        
        first_char = word[0]
        first_upper = turkish_lower_to_upper.get(first_char, first_char.upper())
        
        return first_upper + word[1:].lower()

    def spell_suggest(self, word: str, max_suggestions: int = 5) -> tuple[list[str], str]:
        """Yanlış kelime için önerileri döner."""
        try:
            resp = self.spell.Suggest(spellcheck.SuggestionRequest(word=word, max_suggestions=max_suggestions))
            return list(resp.suggestions), getattr(resp, "error", "")
        except Exception as e:
            return [], str(e)

    def spell_check_batch(self, words: list[str]) -> tuple[list[dict], str]:
        """
        Çoklu kelime kontrolü.
        Her sonuç: {'word':..., 'is_correct': bool, 'suggestions':[...]}
        """
        try:
            resp = self.spell.CheckBatch(spellcheck.BatchSpellCheckRequest(words=words))
            results = []
            for r in resp.results:
                results.append({
                    "word": r.word,
                    "is_correct": r.is_correct,
                    "suggestions": list(r.suggestions)
                })
            return results, getattr(resp, "error", "")
        except Exception as e:
            return [], str(e)

# Son ekler (isim tamlayıcı/iyelik/yalın hâle geçişte pratik)
SUFFIX_TOKENS = {
    "i","ı","u","ü","yi","yı","yu","yü",
    "si","sı","su","sü",
    "in","ın","un","ün",
    "e","a","de","da","te","ta","den","dan","ten","tan"
}

def split_compound(token: str, lex: ConceptLexicon):
    """
    token'ı, kavram sözlüğündeki kelimelerle kapsayacak şekilde böler.
    Hedef: fazproblemi -> ["faz", "problemi"] (problem + i birleşsin)
    DP yaklaşımı: token[0:k] için en iyi bölmeyi bul.
    """
    t = tr_lower(token)
    n = len(t)

    # DP: her pozisyona kadar en iyi (skor, parçalar)
    # skor = eşleşen kavram parça sayısı (maksimize)
    best = [(-1, [])] * (n + 1)
    best[0] = (0, [])

    for i in range(n):
        score_i, parts_i = best[i]
        if score_i < 0:
            continue

        # 1) Normal kavram eşleşmesi (i..j)
        for j in range(i+1, n+1):
            piece = t[i:j]
            if piece in lex.words_lc:
                # parça geçerli bir kavram
                if best[j][0] < score_i + 1:
                    best[j] = (score_i + 1, parts_i + [piece])

        # 2) Sonda ek yakalayarak birleştir (i..k) + ek (k..j)
        #   ör: "problemi" => "problem" + "i" (i..k) kavram, (k..j) ek
        for k in range(i+1, n+1):
            stem = t[i:k]
            if stem in lex.words_lc:
                # tek harfli/çift harfli ek aralığı dene
                for j in (k+1, k+2, k+3):
                    if j <= n:
                        suf = t[k:j]
                        if suf in SUFFIX_TOKENS:
                            # "stem+suf" tek token olacak (problemi)
                            merged = stem + suf
                            # ama kavrama en yakın hali "stem" olduğu için listede "merged" dursun
                            if best[j][0] < score_i + 1:
                                best[j] = (score_i + 1, parts_i + [merged])

    score_n, parts_n = best[n]
    # En az 2 parça ve toplam uzunluk ilerleme sağlıyorsa kabul et
    if score_n >= 2 and "".join(parts_n) == t:
        # orijinal büyük/küçük harf hassasiyeti için parçaları raw'a eşle
        # (burada küçük harfle döndürüyoruz; istenirse capitalize mantığı eklenir)
        return parts_n
    return None


def rebuild_text_from_tokens(tokens):
    """
    tokens: [(token, type), ...]  -> tek metin
    Noktalama öncesi boşluğu temizler.
    """
    txt = " ".join(t for t, tp in tokens)
    # boşluk + noktalama -> noktalama
    txt = re.sub(r"\s+([,.;:!?])", r"\1", txt)
    return txt

def preprocess_text_with_concepts(
    text: str,
    zc,
    concept_dict: set[str] | None = None,
    do_not_correct: set[str] | None = None,
    max_edit_dist: int = 2,
    enable_compound_split: bool = True,
    keep_types: set[str] | None = None,
):
    """
    Zemberek tabanlı boru hattı + kavram sözlüğüyle düzeltme ve birleşik kelime bölme.
    """
    keep_types = keep_types or {"Word", "Number", "Punctuation"}
    do_not_correct = do_not_correct or set()
    lex = ConceptLexicon(concept_dict or set())

    # 1) temel temizlik
    raw = text
    cleaned = zc._basic_unicode_clean(raw)
    lang_id = zc.detect_lang(cleaned)

    # 2) Zemberek normalizasyonu
    normalized, norm_err = zc.normalize_text(cleaned)

    # 3) Tokenize (normalize sonrası)
    toks = zc.tokenize(normalized)

    concept_changes = []     # (orig -> concept_candidate)
    compound_changes = []    # (orig -> [pieces])

    final_tokens = []
    for tok, tp in toks:
        if tp != "Word":
            final_tokens.append((tok, tp))
            continue

        # Do-not-correct: hiç dokunma
        if tok in do_not_correct or tr_lower(tok) in {tr_lower(x) for x in do_not_correct}:
            final_tokens.append((tok, tp))
            continue

        # 3a) Birleşik kelime bölme
        if enable_compound_split and len(tok) > 6 and len(lex.words_lc) > 0:
            pieces = split_compound(tok, lex)
            if pieces:
                # pieces küçük harf; istersen burada biçem uygula
                compound_changes.append((tok, pieces))
                # parçaları token listesine ekle (Word olarak)
                for p in pieces:
                    final_tokens.append((p, "Word"))
                continue  # split yaptıysak kavram-spell aşamasına gerek yok

        # 3b) Kavram sözlüğüyle yazım düzeltme (yakın aday)
        if len(lex.words_lc) > 0 and not lex.has(tok):
            cand = lex.nearest(tok, max_edit_dist)
            if cand is not None:
                concept_changes.append((tok, cand))
                final_tokens.append((cand, "Word"))
                continue

        # Değiştirmeden bırak
        final_tokens.append((tok, tp))

    # 4) Metni yeniden kur ve lemma al
    final_text = rebuild_text_from_tokens(final_tokens)
    morph = zc.analyze_sentence(final_text)
    lemmas = []
    for item in morph:
        ls = item.get("lemmas", [])
        if ls:
            lemmas.extend(ls)
        else:
            tok = item.get("token", "")
            lemmas.append(tr_lower(tok))

    # benzersiz, sıralı
    seen, lemma_seq = set(), []
    for l in lemmas:
        if l not in seen:
            seen.add(l)
            lemma_seq.append(l)

    return {
        "lang": lang_id,
        "raw": raw,
        "cleaned": cleaned,
        "normalized": normalized,
        "normalization_error": norm_err,
        "tokens": final_tokens,
        "concept_changes": concept_changes,   # [('sigor', 'sigorta'), ...]
        "compound_changes": compound_changes, # [('fazproblemi', ['faz','problemi']), ...]
        "final_text": final_text,
        "lemmas": lemma_seq,
        "lemma_text": " ".join(lemma_seq),
    }


