"""Download approved public artifacts locally when the GPU server cannot reach the source."""
import argparse, concurrent.futures, hashlib, json, pathlib, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS=[
 {"name":"vllm-cu129","url":"https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl","path":"cache/wheels/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl","bytes":559420981,"sha256":"9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b"},
 {"name":"spider-official","url":"https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&authuser=0&confirm=t","path":"data/raw/spider/spider_data.zip","bytes":205800266,"sha256":"00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"}
]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def fetch(item):
 target=ROOT/item["path"];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  if target.stat().st_size==item["bytes"] and sha(target)==item["sha256"]:return {**item,"verified":True}
  raise RuntimeError("Existing artifact does not match: "+str(target))
 part=target.with_name(target.name+".partial")
 for attempt in range(4):
  try:
   offset=part.stat().st_size if part.exists() else 0
   req=urllib.request.Request(item["url"],headers={"User-Agent":"Mozilla/5.0",**({"Range":f"bytes={offset}-"} if offset else {})})
   with urllib.request.urlopen(req,timeout=90) as response:
    append=bool(offset and response.status==206)
    if not append:offset=0
    last=0
    with part.open("ab" if append else "wb") as out:
     for b in iter(lambda:response.read(2*1024*1024),b""):
      out.write(b);offset+=len(b)
      if offset>item["bytes"]:raise RuntimeError("Oversized response")
      if time.time()-last>10:print(item["name"],offset,"/",item["bytes"],flush=True);last=time.time()
   if part.stat().st_size!=item["bytes"]:raise RuntimeError("Incomplete artifact")
   if sha(part)!=item["sha256"]:raise RuntimeError("SHA256 mismatch")
   part.replace(target)
   print("VERIFIED",item["name"],flush=True)
   return {**item,"verified":True}
  except Exception as exc:
   print("RETRY",item["name"],repr(exc),flush=True)
   if attempt==3 or "SHA256" in str(exc):raise
   time.sleep(2**attempt)
if __name__=="__main__":
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,ARTIFACTS))
 out=ROOT/".local/artifacts-download-receipt.json";out.parent.mkdir(exist_ok=True)
 out.write_text(json.dumps(records,indent=2),encoding="utf-8")
 print("ARTIFACTS_COMPLETE",flush=True)
