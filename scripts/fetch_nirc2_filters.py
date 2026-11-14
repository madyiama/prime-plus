"""Fetch Keck/KAI measured NIRC2 filter transmissions to a local directory.

An explicit download step, never executed implicitly by reconstruction.
Upstream commit and individual Git blob hashes are pinned for provenance.

Example:
  uv run python scripts/fetch_nirc2_filters.py --filters Kp Ks K
"""
from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib

REPO="Keck-DataReductionPipelines/KAI"
COMMIT="bbb37cdebf86def7e654bfd1541d359a881deeef"
BLOBS={
    "J":"04fa7f5cd4bd02cecc860dee75c63f0042df7ebc",
    "H":"7cb468f3f45c5512756264d20161e9abf13b2ef3",
    "K":"1bf7d377e64cff9fe85384a2ee78b4ba313abd81",
    "Ks":"724365c1d6502308de879238318a46a61719cb48",
    "Kp":"0717e754e97965d3e420cc68c9d1567a953038b5",
    "Lp":"7c4abc095b795b57f6a0e33bce609ca1636d6a24",
    "Ms":"fef07d5a0067e7bdd041c601a14bd34f059a7c4c",
    "Brgamma":"99104dd9179c3469ced13d4a7ed94365cef73bb8",
    "FeII":"f58a6a46bd87fd8ef886ed6ab06b1c9c221dc855",
    "Hcont":"98dc222fa89c44cb46bd6152ac48efee728ad83a",
    "Kcont":"b55b1beca007ff54034d186547f6a2aa88721e75",
}

def git_blob_hash(payload):
    prefix=f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix+payload).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filters",nargs="+",choices=sorted(BLOBS),
                        default=["Kp","Ks","K"])
    parser.add_argument("--output",type=Path,
                        default=Path("data/nirc2_filters"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name in args.filters:
        url=(f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
             f"/kai/filters/{name}.dat")
        with urlopen(url,timeout=30) as response:
            payload=response.read()
        if git_blob_hash(payload)!=BLOBS[name]:
            raise ValueError(f"{name}: upstream KAI source hash mismatch")
        target=args.output/f"{name}.dat"
        target.write_bytes(payload)
        print(name,target,"verified KAI blob",BLOBS[name])

if __name__=="__main__":
    main()
