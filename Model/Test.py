# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:09:25 2025

@author: vural.bayrakli
"""


sample = "onun guclu oldugunu belırttı"

resp = zemb.morph.AnalyzeSentence(morph_pb.SentenceAnalysisRequest(input=sample))

resp

zemb.analyze_sentence("tüketici")

zemb.spell_suggest("ızole")[0]
zemb.spell_check("alaşehir")
zemb.norm.Normalize(norm_pb.NormalizationRequest(input=sample))
zemb.normalize_text(sample)
zc.tokenize(sample)

