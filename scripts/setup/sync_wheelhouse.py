"""Incrementally copy completed, verified wheels to this project's GPU server."""
import base64,json,os,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
DIR=ROOT/"cache/serving-wheelhouse"
HOST=os.environ["AGENT_SSH_HOST"]
REMOTE=os.environ.get("AGENT_REMOTE_WHEELHOUSE","agent/cache/wheelhouse")
if not re.fullmatch(r"/?[A-Za-z0-9_./-]+",REMOTE) or ".." in pathlib.PurePosixPath(REMOTE).parts:raise ValueError("Expected a simple project wheelhouse path")
FLAGS=getattr(subprocess,"CREATE_NO_WINDOW",0)
probe="""import pathlib,json,os
p=pathlib.Path(REMOTE_PATH);p.mkdir(parents=True,exist_ok=True)
v=p.parent/'incoming/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
if v.exists() and not (p/v.name).exists():os.link(v,p/v.name)
print(json.dumps({x.name:x.stat().st_size for x in p.glob('*.whl')}))
"""
probe=probe.replace("REMOTE_PATH",repr(REMOTE))
encoded=base64.b64encode(probe.encode()).decode()
command='python3 -c "import base64;exec(base64.b64decode(\''+encoded+'\'))"'
r=subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=12",HOST,command],capture_output=True,text=True,check=True,creationflags=FLAGS)
existing=json.loads(r.stdout)
todo=[p for p in DIR.glob("*.whl") if existing.get(p.name)!=p.stat().st_size]
print("WHEELS_TO_UPLOAD",len(todo),"BYTES",sum(p.stat().st_size for p in todo),flush=True)
for i in range(0,len(todo),20):
 chunk=todo[i:i+20]
 subprocess.run(["scp","-o","BatchMode=yes","-o","ConnectTimeout=12",*[str(p) for p in chunk],HOST+":"+REMOTE+"/"],check=True,creationflags=FLAGS)
 print("UPLOADED",min(i+20,len(todo)),"/",len(todo),flush=True)
