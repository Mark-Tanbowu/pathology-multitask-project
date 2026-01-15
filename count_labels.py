from collections import Counter
import csv
from pathlib import Path
for split in ['train','val']:
    path = Path('data') / split / 'labels.csv'
    cnt = Counter()
    with path.open(newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cnt[int(row[1])] += 
    total = cnt[0] + cnt[1]
