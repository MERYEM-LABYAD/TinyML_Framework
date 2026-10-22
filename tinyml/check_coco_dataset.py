import json

PATH = r"E:\Stage2\IDID DATASET\testing_data.json"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("CATEGORIES")
for c in data["categories"]:
    print(c)

print("\nCATEGORY IDS DES ANNOTATIONS")
ids = sorted(set(a["category_id"] for a in data["annotations"]))
print(ids)

print("\nNOMBRE ANNOTATIONS PAR CATEGORY_ID")
for cid in ids:
    count = sum(
        1
        for a in data["annotations"]
        if a["category_id"] == cid
    )
    print(cid, count)

print("\nPREMIERE ANNOTATION")
print(data["annotations"][0])