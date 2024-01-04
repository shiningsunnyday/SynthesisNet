import argparse
import os
import pickle
from synnet.utils.data_utils import Program
from synnet.config import PRODUCT_DIR, PRODUCT_JSON
import json
import uuid
from tqdm import tqdm
from multiprocessing import Pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--ncpu', type=int, default=0)
    args = parser.parse_args()
    json_files = set()
    for f in os.listdir(args.dir):
        if f.split('/')[-1] in ['2.pkl']:
            data=pickle.load(open(os.path.join(args.dir, f), 'rb'))
            for d in list(data):
                if args.ncpu:
                    with Pool(args.ncpu) as p:
                        data[d] = p.map(Program.migrate, tqdm(data[d]))
                else:
                    for i, p in tqdm(enumerate(data[d])):
                        data[d][i] = Program.migrate(p)
                for p in data[d]:
                    for fpath in p.product_map.fpaths.values():
                        assert os.path.exists(fpath)
            breakpoint()
            pickle.dump(data, open(os.path.join(args.dir, f), 'wb+'))
