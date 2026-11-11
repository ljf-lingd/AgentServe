"""Fetch a Linux-resolved wheel set from official indexes without host-marker re-resolution."""
import concurrent.futures,hashlib,html.parser,json,os,pathlib,shutil,threading,time,urllib.parse,urllib.request
from pip._vendor import requests
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import cpython_tags,compatible_tags
from pip._vendor.packaging.utils import parse_wheel_filename
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/"cache/serving-wheelhouse"
platforms=[f"manylinux_2_{x}_x86_64" for x in range(35,16,-1)]+["manylinux2014_x86_64","manylinux2010_x86_64","manylinux1_x86_64","linux_x86_64"]
tags=list(cpython_tags((3,11),abis=["cp311"],platforms=platforms))+list(compatible_tags((3,11),interpreter="cp311",platforms=platforms))
rank={tag:i for i,tag in reversed(list(enumerate(tags)))}
LOCK=threading.Lock();progress={};started=time.time()
LOCAL=threading.local()
class Response:
 def __init__(self,response):
  self.response=response
  response.raise_for_status()
  self.status=response.status_code
 def __enter__(self):return self
 def __exit__(self,*args):self.response.close()
 def read(self,n=-1):
  return self.response.content if n<0 else self.response.raw.read(n,decode_content=True)
def request(url,**kwargs):
 if not hasattr(LOCAL,"session"):LOCAL.session=requests.Session()
 timeout=kwargs.pop("timeout",90)
 headers={"User-Agent":"AgentServe-offline-setup/1.0","Accept-Encoding":"identity",**kwargs.pop("headers",{})}
 return Response(LOCAL.session.get(url,headers=headers,stream=True,timeout=(20,timeout)))
def json_get(url):
 with request(url,timeout=20) as r:return json.load(r)
def wheel_rank(filename):
 try:
  _,_,_,ts=parse_wheel_filename(filename)
  return min((rank[t] for t in ts if t in rank),default=10**9)
 except Exception:return 10**9
class Links(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=="a":
   d=dict(attrs)
   if "href" in d:self.links.append(d["href"])
def metadata(req):
 for attempt in range(4):
  try:
   version=next(iter(req.specifier)).version
   cache=ROOT/".local/wheel-metadata"/(req.name+"--"+version+".json")
   if cache.exists():return json.loads(cache.read_text())
   if req.name in ["torch","torchvision","torchaudio"] and "+cu129" in version:
    url=f"https://download.pytorch.org/whl/cu129/{req.name}/"
    with request(url,timeout=20) as r:text=r.read().decode()
    parser=Links();parser.feed(text);candidates=[]
    for link in parser.links:
     filename=urllib.parse.unquote(urllib.parse.urlparse(link).path.rsplit("/",1)[-1])
     try:_,v,_,_=parse_wheel_filename(filename)
     except Exception:continue
     if str(v)==version and wheel_rank(filename)<10**9:
      digest=urllib.parse.parse_qs(urllib.parse.urlparse(link).fragment).get("sha256",[None])[0]
      candidates.append({"filename":filename,"url":urllib.parse.urljoin(url,link.split("#")[0]),"sha256":digest,"bytes":None})
   else:
    data=json_get("https://pypi.org/pypi/"+urllib.parse.quote(req.name)+"/"+urllib.parse.quote(version)+"/json")
    candidates=[{"filename":x["filename"],"url":x["url"],"sha256":x["digests"]["sha256"],"bytes":x["size"]} for x in data["urls"] if x["packagetype"]=="bdist_wheel" and wheel_rank(x["filename"])<10**9]
   if not candidates:raise RuntimeError("No compatible Linux wheel for "+str(req))
   selected=min(candidates,key=lambda x:wheel_rank(x["filename"]))
   result={"requirement":str(req),**selected}
   cache.parent.mkdir(parents=True,exist_ok=True);cache.write_text(json.dumps(result))
   print("METADATA_OK",req.name,flush=True)
   return result
  except Exception as exc:
   print("METADATA_RETRY",str(req),attempt+1,str(exc)[:100],flush=True)
   if attempt==3:raise
   time.sleep(attempt+1)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def update(name,size,complete=False):
 with LOCK:
  progress[name]={"bytes":size,"complete":complete}
  report={"state":"downloading","elapsed_seconds":round(time.time()-started),"complete":sum(x["complete"] for x in progress.values()),"downloaded_bytes":sum(x["bytes"] for x in progress.values()),"files":progress}
  tmp=ROOT/".local/wheelhouse-progress.tmp";tmp.write_text(json.dumps(report));tmp.replace(ROOT/".local/wheelhouse-progress.json")
def fetch_ranges(item,target,part):
 total=item.get("bytes")
 if not total:
  with request(item["url"],headers={"Range":"bytes=0-0"}) as r:
   if r.status!=206:raise RuntimeError("Range request not supported")
   total=int(r.response.headers["Content-Range"].rsplit("/",1)[1])
 item={**item,"bytes":total}
 base=part.stat().st_size if part.exists() else 0
 folder=OUT.parent/"wheelhouse-parts"/item["filename"];folder.mkdir(parents=True,exist_ok=True)
 ranges=[(start,min(start+32*1024*1024,total)-1) for start in range(base,total,32*1024*1024)]
 positions={start:0 for start,_ in ranges};cl=threading.Lock()
 def piece(pair):
  start,end=pair;dest=folder/(str(start)+"-"+str(end)+".bin")
  if dest.exists() and dest.stat().st_size==end-start+1:
   positions[start]=end-start+1;return dest
  tmp=dest.with_suffix(".tmp")
  for attempt in range(4):
   try:
    req_start=start+(tmp.stat().st_size if tmp.exists() else 0)
    if req_start<=end:
     with request(item["url"],headers={"Range":f"bytes={req_start}-{end}"}) as r:
      if r.status!=206 or not r.response.headers.get("Content-Range","").startswith(f"bytes {req_start}-"):raise RuntimeError("Incorrect range response")
      offset=req_start-start;last=0
      with tmp.open("ab" if offset else "wb") as out:
       for block in iter(lambda:r.read(2*1024*1024),b""):
        out.write(block);offset+=len(block)
        with cl:
         positions[start]=offset;received=base+sum(positions.values())
        if time.time()-last>3:update(item["filename"],received);last=time.time()
    if tmp.stat().st_size!=end-start+1:raise RuntimeError("Incomplete range")
    tmp.replace(dest);return dest
   except Exception:
    if attempt==3:raise
    time.sleep(attempt+1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:chunks=list(pool.map(piece,ranges))
 assembled=target.with_name(target.name+".assembling")
 with assembled.open("wb") as out:
  if base:
   with part.open("rb") as f:shutil.copyfileobj(f,out,8*1024*1024)
  for chunk in chunks:
   with chunk.open("rb") as f:shutil.copyfileobj(f,out,8*1024*1024)
 actual=sha(assembled)
 if assembled.stat().st_size!=total or (item.get("sha256") and actual!=item["sha256"]):raise RuntimeError("Assembled wheel checksum mismatch")
 assembled.replace(target)
 if part.exists():part.unlink()
 for chunk in chunks:chunk.unlink()
 folder.rmdir()
 update(item["filename"],total,True)
 print("VERIFIED_RANGES",item["filename"],flush=True)
 return {**item,"actual_sha256":actual}

def fetch(item):
 target=OUT/item["filename"];part=target.with_name(target.name+".partial")
 if target.exists():
  actual=sha(target)
  if item.get("sha256") and actual!=item["sha256"]:raise RuntimeError("Existing wheel hash mismatch")
  update(item["filename"],target.stat().st_size,True)
  return {**item,"actual_sha256":actual,"bytes":target.stat().st_size}
 if (item.get("bytes") or 0)>128*1024*1024 or item["filename"].startswith("torch-"):
  return fetch_ranges(item,target,part)
 for attempt in range(4):
  try:
   offset=part.stat().st_size if part.exists() else 0
   with request(item["url"],headers={"Range":f"bytes={offset}-"} if offset else {}) as r:
    append=bool(offset and r.status==206)
    if not append:offset=0
    last=0
    with part.open("ab" if append else "wb") as f:
     for b in iter(lambda:r.read(2*1024*1024),b""):
      f.write(b);offset+=len(b)
      if time.time()-last>5:update(item["filename"],offset);last=time.time()
   if item.get("bytes") and part.stat().st_size!=item["bytes"]:raise RuntimeError("Incomplete wheel")
   actual=sha(part)
   if item.get("sha256") and actual!=item["sha256"]:raise RuntimeError("Wheel checksum mismatch: "+item["filename"])
   part.replace(target);update(item["filename"],target.stat().st_size,True)
   print("VERIFIED_WHEEL",item["filename"],flush=True)
   return {**item,"actual_sha256":actual,"bytes":target.stat().st_size}
  except Exception:
   if attempt==3:raise
   time.sleep(attempt+1)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lines=(ROOT/".local/setup-linux.download.txt").read_text().splitlines()
 reqs=[Requirement(x) for x in lines if x.strip() and not x.lstrip().startswith("#") and not x.strip().endswith(".whl")]
 source=ROOT/"cache/wheels/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
 target=OUT/source.name
 if not target.exists():
  try:os.link(source,target)
  except OSError:shutil.copy2(source,target)
 reqs=[r for r in reqs if r.name!="vllm"]
 print("FETCH_METADATA",len(reqs),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:items=list(pool.map(metadata,reqs))
 (ROOT/".local/wheelhouse-source-manifest.json").write_text(json.dumps(items,indent=2))
 print("DOWNLOAD_WHEELS",len(items),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(fetch,items))
 records.append({"filename":source.name,"requirement":"vllm==0.32.0+cu129","actual_sha256":sha(target),"bytes":target.stat().st_size,"url":"https://github.com/vllm-project/vllm/releases/tag/v0.32.0"})
 (ROOT/".local/wheelhouse-manifest.json").write_text(json.dumps(records,indent=2))
 print("WHEELHOUSE_COMPLETE",len(records),sum(x["bytes"] for x in records),flush=True)
if __name__=="__main__":main()
