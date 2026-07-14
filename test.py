import os

folders = sorted(os.listdir("raw_data/Chars74K/English/Fnt"))

print(folders[:30])
print("Total:", len(folders))