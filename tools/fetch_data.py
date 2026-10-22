"""Obtain only the unchanged course data; verify every byte against the release manifest."""
import argparse, hashlib, json, shutil, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-dir',type=Path,help='Copy an existing official code/data directory instead of downloading')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
    args=ap.parse_args();manifest=json.loads((ROOT/'DATA_SOURCES.json').read_text())
    args.data_dir.mkdir(parents=True,exist_ok=True)
    for rel,entry in manifest['files'].items():
        out=args.data_dir/rel
        if out.exists():
            if digest(out)!=entry['sha256']:raise RuntimeError('Existing data differs; preserve it and choose another --data-dir: '+str(out))
            continue
        out.parent.mkdir(parents=True,exist_ok=True)
        temporary=out.with_name(out.name+'.download')
        if temporary.exists():raise FileExistsError('Remove or inspect prior partial download: '+str(temporary))
        if args.source_dir:
            source=args.source_dir/rel
            if digest(source)!=entry['sha256']:raise RuntimeError('Source differs: '+str(source))
            shutil.copyfile(source,temporary)
        else:
            url=f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['ref']}/{manifest['prefix']}/{rel}"
            request=urllib.request.Request(url,headers={'User-Agent':'DASE7506-MP2-data-setup'})
            with urllib.request.urlopen(request,timeout=180) as response,temporary.open('wb') as stream:
                shutil.copyfileobj(response,stream)
        if temporary.stat().st_size!=entry['bytes'] or digest(temporary)!=entry['sha256']:
            raise RuntimeError('Data hash mismatch; inspect the partial file: '+str(temporary))
        temporary.rename(out)
        print('Verified '+rel,flush=True)
    print(json.dumps({'data_files_verified':len(manifest['files']),'data_directory':str(args.data_dir)}))
if __name__=='__main__':main()

