"""Regenerate a session heatmap from its gaze/posture CSV log."""
import argparse
import csv
import json
from pathlib import Path

from gaze_visualization import save_gaze_visualization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    args=parser.parse_args()
    setup=json.loads((args.session/'setup.json').read_text(encoding='utf-8'))
    _,_,width,height=setup['monitor']
    records=[]
    with (args.session/'gaze_posture_embedding.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['gaze_x_px'] and row['gaze_y_px']:
                records.append((float(row['timestamp']),float(row['gaze_x_px']),float(row['gaze_y_px'])))
    output,summary=save_gaze_visualization(records,int(width),int(height),args.session)
    print(output)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
