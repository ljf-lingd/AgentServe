"""Create isolated project environments using the existing Miniconda manager."""
import json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[2]
CONDA_BASE=pathlib.Path(os.environ.get("AGENT_CONDA_BASE",str(pathlib.Path.home()/"miniconda3")))
CONDA=CONDA_BASE/"bin/conda"
ENVROOT=CONDA_BASE/"envs"
env=os.environ.copy()
env["CONDA_PLUGINS_AUTO_ACCEPT_TOS"]="yes"
env["CONDA_PKGS_DIRS"]=str(CONDA_BASE/"pkgs")
env["PIP_CACHE_DIR"]=str(ROOT/"cache/pip")
env["PIP_DISABLE_PIP_VERSION_CHECK"]="1"
env["PIP_DEFAULT_TIMEOUT"]="90"
env["UV_CACHE_DIR"]=str(ROOT/"cache/uv")
env["UV_HTTP_TIMEOUT"]="60"
def run(args):
    print("RUN",json.dumps([str(a) for a in args]),flush=True)
    subprocess.run([str(a) for a in args],env=env,check=True)
def main():
    for name in ["agent","agent-vllm"]:
        if not (ENVROOT/name/"bin/python").exists():
            run([CONDA,"create","-n",name,"python=3.11","pip","-y","--override-channels","-c","https://repo.anaconda.com/pkgs/main"])
    py=ENVROOT/"agent/bin/python"
    packages=["fastapi","uvicorn","pydantic","httpx","aiosqlite","pyyaml","pytest","pytest-asyncio","matplotlib","pyarrow","ruff","gdown","sqlparse","nltk"]
    run(["/snap/bin/uv","pip","install","--python",py,"--no-index","--find-links",ROOT/"cache/wheelhouse",*packages])
    run([py,"-m","pip","check"])
    freeze=subprocess.check_output([str(py),"-m","pip","freeze"],env=env,text=True)
    (ROOT/"envs/agent.requirements.lock.txt").write_text(freeze)
    (ROOT/"logs/setup-envs.json").write_text(json.dumps({"state":"app_complete","app_python":str(py),"serving_python":str(ENVROOT/"agent-vllm/bin/python"),"packages":packages},indent=2))
if __name__=="__main__":
    try:main()
    except Exception as exc:
        (ROOT/"logs/setup-envs.json").write_text(json.dumps({"state":"failed","error":repr(exc)}));raise
