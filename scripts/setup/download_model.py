"""Download the approved Qwen snapshot from ModelScope and verify every file."""
import concurrent.futures, hashlib, json, os, pathlib, shutil, threading, time
import urllib.parse, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2]
DEST=ROOT/"models/Qwen2.5-7B-Instruct"
STATUS=ROOT/"logs/setup-model.json"
LOCK=threading.Lock()
progress={}
started=time.time()
def status(state, **extra):
    with LOCK:
        obj={"state":state,"elapsed_seconds":round(time.time()-started),"files":dict(progress),**extra}
        tmp=STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj,indent=2),encoding="utf-8")
        tmp.replace(STATUS)
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def download(meta):
    name=meta["Path"]
    if pathlib.PurePosixPath(name).name!=name: raise ValueError("Unexpected model path")
    target=DEST/name
    expected=int(meta["Size"]); expected_sha=meta["Sha256"]
    if target.exists():
        if target.stat().st_size==expected and digest(target)==expected_sha:
            progress[name]=expected
            return {"file":name,"bytes":expected,"sha256":expected_sha,"verified":True}
        raise RuntimeError("Existing target does not match manifest: "+name)
    part=target.with_name(target.name+".partial")
    query=urllib.parse.urlencode({"Revision":meta["Revision"],"FilePath":name})
    url="https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-7B-Instruct/repo?"+query
    for attempt in range(5):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset>expected:raise RuntimeError("Partial file larger than expected")
            if offset<expected:
                req=urllib.request.Request(url,headers={"User-Agent":"AgentServe-setup/1.0",**({"Range":f"bytes={offset}-"} if offset else {})})
                with urllib.request.urlopen(req,timeout=60) as response:
                    append=offset>0 and response.status==206
                    if offset and not append:offset=0
                    last_update=0
                    with part.open("ab" if append else "wb") as out:
                        while True:
                            chunk=response.read(2*1024*1024)
                            if not chunk:break
                            if shutil.disk_usage(ROOT).free<50*2**30:raise RuntimeError("Disk reserve below 50 GiB")
                            out.write(chunk);offset+=len(chunk)
                            if offset>expected:raise RuntimeError("Response exceeds expected model file size")
                            progress[name]=offset
                            if time.time()-last_update>5:
                                status("downloading");last_update=time.time()
            if part.stat().st_size!=expected:raise RuntimeError("Incomplete response")
            actual_sha=digest(part)
            if actual_sha!=expected_sha:raise RuntimeError("SHA256 mismatch: "+name)
            part.replace(target)
            print("VERIFIED",name,expected,flush=True)
            progress[name]=expected
            return {"file":name,"bytes":expected,"sha256":actual_sha,"source_file_revision":meta["Revision"],"verified":True}
        except Exception as exc:
            print("RETRY",name,attempt+1,type(exc).__name__,str(exc)[:180],flush=True)
            if "SHA256 mismatch" in str(exc) or attempt==4:raise
            time.sleep(min(2**attempt,16))
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    raw=json.loads((ROOT/"configs/models/qwen2.5-7b-instruct.manifest.json").read_text())
    metas=[m for m in raw["Data"]["Files"] if not m.get("IsDir") and m["Name"]!=".gitattributes"]
    status("starting")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files=list(pool.map(download,metas))
    manifest={"model_id":"Qwen/Qwen2.5-7B-Instruct","source":"ModelScope","source_branch":"master","snapshot_identity":"per-file revisions and SHA256 from saved source manifest","files":files,"total_bytes":sum(f["bytes"] for f in files)}
    (DEST/"DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    index=json.loads((DEST/"model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        if not (DEST/name).is_file():raise RuntimeError("Missing indexed shard")
    status("complete",total_bytes=manifest["total_bytes"])
    print("MODEL_DOWNLOAD_COMPLETE",manifest["total_bytes"],flush=True)
if __name__=="__main__":
    try:main()
    except Exception as exc:
        status("failed",error=repr(exc));raise
