# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:24:50 2025

@author: vural.bayrakli
"""

# dosyanın yolu
filename = "arızalar.txt"

# sonuç sözlüğü
word_dict = {}

with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # sadece '-' ile başlayan satırları al
        if line.startswith("-"):
            # baştaki '-' ve boşlukları temizle
            phrase = line.lstrip("-").strip()
            # kelimelere ayır
            words = phrase.split()
            for w in words:
                # sözlükte varsa artır, yoksa ekle
                word_dict[w] = word_dict.get(w, 0) + 1

# sözlüğü yazdır
for word, count in word_dict.items():
    print(word, ":", count)

word2id = {}
id2word = {}
phrase2ids = {}
ids2phrase = {}

current_id = 1

# Dosyadan satır satır oku
with open("arızalar.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("###"):  # başlık satırlarını atla
            continue
        if line.startswith("-"):
            phrase = line.lstrip("-").strip()
            words = phrase.split()  # örn: ["trafo", "bağlantı", "arıza"]

            ids = []
            for w in words:
                if w not in word2id:
                    word2id[w] = current_id
                    id2word[current_id] = w
                    current_id += 1
                ids.append(word2id[w])

            # phrase -> ids
            phrase2ids[phrase] = ids
            # ids (tuple) -> phrase
            ids2phrase[tuple(ids)] = phrase

# --- Test ---
print("Word2ID:", word2id)
print("Phrase2IDs:", phrase2ids)

# Örnek lookup: [1,2,3] → "trafo bağlantı arızası"
query = [1,2,3]
print("Query sonucu:", ids2phrase.get(tuple(query)))
