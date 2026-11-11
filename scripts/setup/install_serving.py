"""Install the verified latest vLLM CUDA 12.9 wheel in the isolated serving environment."""
import json,os,pathlib,subprocess,traceback,shutil,zipfile,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
PY=pathlib.Path(os.environ.get("VLLM_PYTHON",str(pathlib.Path.home()/"miniconda3/envs/agent-vllm/bin/python")))
WHEEL=ROOT/"cache/incoming/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
env=os.environ.copy()
env["PIP_CACHE_DIR"]=str(ROOT/"cache/pip")
env["PIP_DEFAULT_TIMEOUT"]="120"
env["UV_CACHE_DIR"]=str(ROOT/"cache/uv")
env["UV_HTTP_TIMEOUT"]="60"
env["PIP_DISABLE_PIP_VERSION_CHECK"]="1"
def run(args,**kwargs):
 print("RUN",json.dumps([str(x) for x in args]),flush=True)
 return subprocess.run([str(x) for x in args],env=env,check=True,**kwargs)
def main():
 if shutil.disk_usage(ROOT).free<60*2**30:raise RuntimeError("Insufficient disk reserve before serving install")
 h=hashlib.sha256()
 with WHEEL.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 if h.hexdigest()!="9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b":raise RuntimeError("vLLM wheel hash mismatch")
 with zipfile.ZipFile(WHEEL) as z:
  meta=z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode()
 (ROOT/"envs/vllm-wheel.metadata.txt").write_text(meta,encoding="utf-8")
 run(["/snap/bin/uv","pip","install","--python",PY,"--no-index","--find-links",ROOT/"cache/wheelhouse",str(WHEEL),"torch==2.13.0+cu129"])
 run([PY,"-m","pip","check"])
 run([PY,"-c","import torch,vllm; print('VERSIONS',torch.__version__,torch.version.cuda,vllm.__version__); print('CUDA_AVAILABLE',torch.cuda.is_available()); x=torch.randn(256,256,device='cuda:0'); print('MATMUL_OK',float((x@x).sum())); torch.cuda.synchronize()"])
 freeze=subprocess.check_output([str(PY),"-m","pip","freeze"],env=env,text=True)
 freeze="\n".join("vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b" if line.startswith("vllm @ ") else line for line in freeze.splitlines())+"\n"
 (ROOT/"envs/agent-vllm.requirements.lock.txt").write_text(freeze)
 (ROOT/"logs/setup-serving.json").write_text(json.dumps({"state":"installed_gpu_tensor_verified","requested_vllm":"0.32.0+cu129","requested_torch":"2.13.0+cu129","model_generation_verified":False},indent=2))
if __name__=="__main__":
 try:main()
 except Exception as exc:
  (ROOT/"logs/setup-serving.json").write_text(json.dumps({"state":"failed","error":repr(exc)},indent=2));raise
