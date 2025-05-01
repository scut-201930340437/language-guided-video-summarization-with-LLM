import json
import os

os.makedirs('text_sumy/SumMe', exist_ok=True)

for filename in os.listdir('SumMe_best/'):
    lines = {}
    seg_idx = 1
    with open('SumMe_best/' + filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            if len(line) > 0:  
                lines[seg_idx] = line
                seg_idx = seg_idx + 1
        
    with open("text_sumy/" + 'SumMe' + "/" + "sumy_" + (filename.split(".")[0])[16:] + ".json", 'w') as write_f:
        json.dump(lines, write_f, indent=4, ensure_ascii=False)
        