import argparse
import os
import pickle
from synnet.utils.data_utils import ProductMap, ProductMapLink


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    args = parser.parse_args()
    json_files = set()
    for f in os.listdir(args.dir):
        if f.split('/')[-1] in ['1.pkl', '2.pkl']:
            data=pickle.load(open(os.path.join(args.dir, f), 'rb'))
            for d in data:
                for p in data[d]:
                    if isinstance(p.product_map, ProductMap):
                        json_files.add(p.product_map.fpath)
                    elif isinstance(p.product_map, ProductMapLink):
                        for fpath in p.product_map.fpaths:
                            json_files.add(fpath)
        # if '07d413ef4f8a12b8e2d2a09a012a804e' in f:
        #     data=pickle.load(open(os.path.join(args.dir, f), 'rb'))
        #     breakpoint()
    count = 0
    bad_count = 0
    for f in os.listdir(args.dir):
        if '.json' in f:
            if os.path.join(args.dir, f) not in json_files:
                os.remove(os.path.join(args.dir, f))
