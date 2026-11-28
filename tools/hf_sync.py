"""Move packed NTS training shots to the Hugging Face dataset HF_USER/nts-cycles-train, then free local disk.

The repo is created private; its owner makes it public on the Hub. For each complete shot in ~/nts-data/packed
(meta.json present, unchanged for 3 minutes): upload the folder, list it back from the Hub, and delete the local copy
only if every file is there with the same size. KEEP shots are uploaded but kept. Runs until killed, rechecking every
5 minutes. Log lines: "UPLOADED <shot> <GB>", "KEPT <shot>", "FAIL <shot> ...".

Run (on the 1080): ~/nts-data/kvenv/bin/python tools/hf_sync.py [dataset_card.md]
"""
import os
import shutil
import sys
import time

from huggingface_hub import HfApi

REPO = "HF_USER/nts-cycles-train"
SRC = os.path.expanduser("~/nts-data/packed")
KEEP = {"test_zd_s0"}
api = HfApi()


def complete(d):
    if not os.path.exists(os.path.join(d, "meta.json")):
        return False
    newest = max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
    return time.time() - newest > 180


def remote_sizes(shot):
    out = {}
    try:
        for f in api.list_repo_tree(REPO, path_in_repo=shot, repo_type="dataset", recursive=True):
            if getattr(f, "size", None) is not None:
                out[os.path.basename(f.path)] = f.size
    except Exception:  # the folder doesn't exist yet
        pass
    return out


def main():
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    if len(sys.argv) > 1:
        api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo="README.md", repo_id=REPO, repo_type="dataset",
                        commit_message="dataset card")
    done = set()
    while True:
        for shot in sorted(os.listdir(SRC)):
            d = os.path.join(SRC, shot)
            if shot in done or not os.path.isdir(d) or not complete(d):
                continue
            local = {f: os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)}
            try:
                if remote_sizes(shot) != local:
                    api.upload_folder(folder_path=d, path_in_repo=shot, repo_id=REPO, repo_type="dataset",
                                      commit_message=f"add {shot}")
                if remote_sizes(shot) != local:
                    print(f"FAIL {shot} size mismatch after upload", flush=True)
                    continue
            except Exception as e:  # network or rate limit: retry next round
                print(f"FAIL {shot} {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            gb = sum(local.values()) / 1e9
            if shot in KEEP:
                print(f"KEPT {shot} {gb:.1f}G (uploaded)", flush=True)
            else:
                shutil.rmtree(d)
                print(f"UPLOADED {shot} {gb:.1f}G {time.strftime('%H:%M')}", flush=True)
            done.add(shot)
        time.sleep(300)


if __name__ == "__main__":
    main()
