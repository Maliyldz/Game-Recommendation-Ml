"""
import pandas as pd

df = pd.read_csv("steam_reviews.csv/steam_reviews.csv")

# Sadece İngilizce ('english') olanları filtrele
df_english = df[df['language'] == 'english']

# Filtrelenmiş veriyi yeni bir dosyaya kaydet
df_english.to_csv("steam_reviews_english.csv", index=False)
"""

"""
import pandas as pd
import re

df = pd.read_csv("steam_reviews_english.csv")

# 1. Boş ve boşluk olan yorumları temizle
df_cleaned = df[df['review'].notnull()]
df_cleaned = df_cleaned[df_cleaned['review'].str.strip() != '']

# 2. Sadece sembolden oluşan yorumları temizle
symbol_pattern = re.compile(r'^[^\w\s]+$')
df_cleaned = df_cleaned[~df_cleaned['review'].str.match(symbol_pattern)]

# 3. 3 kelimeden az olan yorumları temizle
df_cleaned = df_cleaned[df_cleaned['review'].str.split().str.len() >= 3]

# 4. İngilizce karakter oranı düşük olan yorumları temizle
def is_mostly_english(text, threshold=0.7):
    english_chars = sum(c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz' for c in text)
    total_chars = len(text)
    if total_chars == 0:
        return False
    return (english_chars / total_chars) >= threshold

df_cleaned = df_cleaned[df_cleaned['review'].apply(is_mostly_english)]

# 5. Anlamsız kelimeleri içeren yorumları temizle
nonsense_pattern = re.compile(r'\b(?:asdf|qwer|sdfg|sdds|lolol|haha|lmao|xd|qweqwe)\b', re.IGNORECASE)
df_cleaned = df_cleaned[~df_cleaned['review'].str.contains(nonsense_pattern)]

# Sonuçları yeni bir dosyaya kaydet
df_cleaned.to_csv("steam_reviews_cleaned.csv", index=False)
"""

"""
#recommended sütununa göre dengeli bir bölme işlemi yapar
import pandas as pd

df = pd.read_csv('steam_reviews_cleaned.csv')

# Her sınıftan alınacak örnek sayısı
samples_per_class = 15000

# recommended = 0 ve 1 olanlardan örnek al
sample_0 = df[df['recommended'] == 0].sample(n=samples_per_class, random_state=42)
sample_1 = df[df['recommended'] == 1].sample(n=samples_per_class, random_state=42)

# Birleştir ve karıştır
balanced_sample = pd.concat([sample_0, sample_1]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Toplam örneklem: {len(balanced_sample)} satır")
print(balanced_sample['recommended'].value_counts())

balanced_sample.to_csv('orneklem_30_bin_dengeli.csv', index=False)
"""

"""
import pandas as pd

df = pd.read_csv("orneklem_30_bin_dengeli.csv")

# Çıkartılacak sütunlar
columns_to_drop = [
    "app_id", "app_name", "review_id", "language",
    "timestamp_created", "timestamp_updated",
    "author.steamid", "author.last_played"
]

# Sütunları çıkart
df_reduced = df.drop(columns=columns_to_drop)

# Yeni dosyaya kaydet
df_reduced.to_csv("reduced_dataset_30_bin_dengeli.csv", index=False)
"""

"""
import pandas as pd

df = pd.read_csv("reduced_dataset_30_bin_dengeli.csv")

# 1. Boolean sütunları bul ve 0/1'e çevir
bool_columns = df.select_dtypes(include=['bool']).columns
df[bool_columns] = df[bool_columns].astype(int)


# 3. Sonuçları kontrol etmek için örnek:
print(f"Boolean sütunlar:\n{df[bool_columns].head()}")

# Boolean sütunları 0/1 yaptıktan sonra kaydet
df.to_csv("processed_dataset_30_bin_dengeli.csv", index=False)
"""
"""
#İsteğe bağlı recommended sütunundaki 1 ve 0 sayısı kontrol edilmek istenirse
import pandas as pd


df = pd.read_csv('processed_dataset_30_bin_dengeli.csv')

# recommended sütunundaki 0 ve 1'leri say
sayilar = df['recommended'].value_counts()

print(sayilar)
"""