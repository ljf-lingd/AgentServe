"""Readiness check only: local GPU generation and tool-call parsing, without an Agent workload."""
import json,os,pathlib,signal,socket,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2]
MODEL=ROOT/"models/Qwen2.5-7B-Instruct"
REPORT=ROOT/"logs/serving-readiness.json"
def request(port,path,payload=None,timeout=60):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(f"http://127.0.0.1:{port}"+path,data=data,headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def main():
 if not (MODEL/"DOWNLOAD_MANIFEST.json").exists():raise RuntimeError("Verified model manifest missing")
 env=os.environ.copy()
 env.update({"CUDA_VISIBLE_DEVICES":"0","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_HOME":str(ROOT/"cache/huggingface"),"VLLM_CACHE_ROOT":str(ROOT/"cache/vllm"),"TRITON_CACHE_DIR":str(ROOT/"cache/triton"),"VLLM_NO_USAGE_STATS":"1","DO_NOT_TRACK":"1"})
 with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
 args=[sys.executable,"-m","vllm.entrypoints.openai.api_server","--model",str(MODEL),"--served-model-name","agent-qwen","--host","127.0.0.1","--port",str(port),"--dtype","bfloat16","--tensor-parallel-size","1","--max-model-len","2048","--gpu-memory-utilization","0.45","--max-num-seqs","2","--enforce-eager","--enable-auto-tool-choice","--tool-call-parser","hermes"]
 log=(ROOT/"logs/vllm/readiness-server.log").open("ab",buffering=0)
 proc=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 report={"state":"starting","server_pid":proc.pid,"port":port,"gpu":"0","gpu_memory_utilization":0.45,"args":args,"demo_created":False,"synthetic_created":False}
 REPORT.write_text(json.dumps(report,indent=2))
 try:
  deadline=time.monotonic()+480
  while time.monotonic()<deadline:
   if proc.poll() is not None:raise RuntimeError(f"vLLM server exited with {proc.returncode}")
   try:request(port,"/v1/models",timeout=2);break
   except Exception:time.sleep(2)
  else:raise TimeoutError("vLLM startup deadline reached")
  plain=request(port,"/v1/chat/completions",{"model":"agent-qwen","messages":[{"role":"user","content":"Reply with the single word READY."}],"temperature":0,"max_tokens":16})
  if not plain.get("choices"):raise RuntimeError("No model generation choices")
  report["plain_generation"]=plain
  tool=request(port,"/v1/chat/completions",{"model":"agent-qwen","messages":[{"role":"user","content":"Call inspect_schema with database_id set to spider."}],"temperature":0,"max_tokens":128,"tools":[{"type":"function","function":{"name":"inspect_schema","description":"Inspect the schema of the selected database.","parameters":{"type":"object","properties":{"database_id":{"type":"string"}},"required":["database_id"],"additionalProperties":False}}}],"tool_choice":{"type":"function","function":{"name":"inspect_schema"}}})
  calls=tool["choices"][0]["message"].get("tool_calls",[])
  if not calls or calls[0]["function"]["name"]!="inspect_schema":raise RuntimeError("Tool-call parser did not return the selected function")
  arguments=json.loads(calls[0]["function"]["arguments"])
  if arguments.get("database_id")!="spider":raise RuntimeError("Unexpected tool-call argument")
  report["tool_call"]=tool
  report["state"]="passed"
  print("SERVING_READINESS_PASSED",flush=True)
 except Exception as exc:
  report["state"]="failed";report["error"]=repr(exc);raise
 finally:
  REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
  log.close()
if __name__=="__main__":main()
