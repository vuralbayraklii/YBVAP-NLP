# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 12:25:18 2025

@author: vural.bayrakli
"""

sample = "yakılmış"

resp = zemb.morph.AnalyzeSentence(morph_pb.SentenceAnalysisRequest(input=sample))

resp

zemb.analyze_sentence("yanık")

zemb.spell_suggest(sample)[0]
zemb.spell_check("cıvata")
zemb.norm.Normalize(norm_pb.NormalizationRequest(input=sample))
zemb.normalize_text(sample)
zc.tokenize(sample)

