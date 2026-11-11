"""Verify and stage the approved Spider archive; do not generate demo or synthetic workloads."""
import hashlib,json,os,pathlib,shutil,sqlite3,zipfile,zlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/"cache/incoming/spider_data.zip"
if not ARCHIVE.exists():ARCHIVE=ROOT/"data/raw/spider/spider_data.zip"
EXPECTED="00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 if sha(ARCHIVE)!=EXPECTED:raise RuntimeError("Spider official archive checksum mismatch")
 raw=ROOT/"data/raw/spider";dbroot=ROOT/"data/databases/spider"
 raw.mkdir(parents=True,exist_ok=True);dbroot.mkdir(parents=True,exist_ok=True)
 official={}
 databases=[]
 with zipfile.ZipFile(ARCHIVE) as z:
  for member in z.infolist():
   parts=pathlib.PurePosixPath(member.filename).parts
   if member.is_dir() or ".." in parts or "__MACOSX" in parts:continue
   basename=parts[-1]
   if basename in ["dev.json","train_spider.json","train_others.json","tables.json"]:
    if basename in official:raise RuntimeError("Duplicate metadata basename")
    data=z.read(member);official[basename]=json.loads(data)
    (raw/basename).write_bytes(data)
   elif basename.endswith(".sqlite") and "database" in parts:
    index=parts.index("database");rel=pathlib.Path(*parts[index+1:])
    target=dbroot/rel
    if not target.resolve().is_relative_to(dbroot.resolve()):raise RuntimeError("Archive escaped target")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
     crc=0
     with target.open("rb") as existing:
      for block in iter(lambda:existing.read(1024*1024),b""):crc=zlib.crc32(block,crc)
     if target.stat().st_size!=member.file_size or crc!=member.CRC:raise RuntimeError("Existing database differs from approved archive")
     databases.append(target)
     continue
    with z.open(member) as src,target.open("xb") as out:shutil.copyfileobj(src,out)
    target.chmod(0o444)
    databases.append(target)
 for key in ["dev.json","train_spider.json","tables.json"]:
  if key not in official:raise RuntimeError("Missing official metadata: "+key)
 train_ids={x["db_id"] for x in official["train_spider.json"]}
 dev_ids={x["db_id"] for x in official["dev.json"]}
 actual_ids={p.parent.name for p in databases}
 if train_ids&dev_ids:raise RuntimeError("Official train/dev database overlap")
 missing=(train_ids|dev_ids)-actual_ids
 if missing:raise RuntimeError("Missing task databases: "+str(sorted(missing)))
 tables_ids={x["db_id"] for x in official["tables.json"]}
 if not (train_ids|dev_ids)<=tables_ids:raise RuntimeError("Schema metadata missing database")
 verified=[]
 for db in databases:
  with sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True) as conn:
   tables=conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
   result=conn.execute("PRAGMA quick_check").fetchall()
   if result!=[("ok",)]:raise RuntimeError("Database quick_check failed: "+str(db))
  verified.append({"db_id":db.parent.name,"path":str(db.relative_to(ROOT)),"bytes":db.stat().st_size,"sha256":sha(db),"tables":len(tables),"quick_check":"ok"})
  print("VERIFIED_DB",db.parent.name,len(tables),flush=True)
 existing=pathlib.Path(os.environ.get("SPIDER_PARQUET_SOURCE",str(raw)))
 for split in ["train","validation"]:
  source=existing/(split+"-00000-of-00001.parquet")
  target=raw/(split+".parquet")
  if source.exists():
   shutil.copy2(source,target)
   print("COPIED_EXISTING_PARQUET",split,sha(target),flush=True)
 final_archive=raw/"spider_data.zip"
 if not final_archive.exists():ARCHIVE.replace(final_archive)
 manifest={"source_url":"https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view","official_page":"https://yale-lily.github.io/spider","archive_sha256":EXPECTED,"license":"CC-BY-SA-4.0","train_questions":len(official["train_spider.json"]),"dev_questions":len(official["dev.json"]),"train_database_count":len(train_ids),"dev_database_count":len(dev_ids),"train_dev_database_overlap":0,"databases":verified,"demo_created":False,"synthetic_created":False,"project_subset_selected":False}
 (raw/"DATASET_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
 (ROOT/"logs/setup-data.json").write_text(json.dumps({"state":"complete","database_count":len(verified),"train_questions":manifest["train_questions"],"dev_questions":manifest["dev_questions"]},indent=2))
 print("SPIDER_READY",len(verified),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as exc:
  (ROOT/"logs/setup-data.json").write_text(json.dumps({"state":"failed","error":repr(exc)}));raise
