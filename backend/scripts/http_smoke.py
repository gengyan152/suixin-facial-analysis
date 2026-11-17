"""Boot the actual single-worker server, wait for real warmup, POST/GET/DELETE."""
from pathlib import Path
import sys,subprocess,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import ROOT,WEIGHTS
import requests


def main():
    report={}
    started=time.monotonic()
    with (ROOT/'http_smoke_server.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','18000','--workers','1','--no-access-log'],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        session=requests.Session();session.trust_env=False
        base='http://127.0.0.1:18000'
        try:
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError('server exited; inspect http_smoke_server.log')
                try:
                    response=session.get(base+'/api/v1/health',timeout=3)
                    health=response.json()
                except requests.ConnectionError:
                    time.sleep(.3);continue
                if health['status']=='degraded': raise RuntimeError(health)
                if health['warmed_up']: break
                time.sleep(.3)
            else: raise TimeoutError('server warmup')
            report['startup_seconds']=time.monotonic()-started
            report['health']=health
            response=session.post(base+'/api/v1/tasks',data=(WEIGHTS/'warmup.jpg').read_bytes(),headers={'Content-Type':'application/octet-stream'},timeout=15)
            response.raise_for_status()
            task=response.json()
            stages=[];deadline=time.monotonic()+110
            while time.monotonic()<deadline:
                result=session.get(base+'/api/v1/tasks/'+task['task_id'],timeout=3).json()
                if not stages or stages[-1]!=result['status']: stages.append(result['status'])
                if result['status'] in {'completed','partial_error','rejected'}: break
                time.sleep(.05)
            assert result['status']=='completed',result
            report['stages']=stages
            report['result']=result
            response=session.get(base+result['original_image_url'],timeout=3)
            assert response.status_code==200 and response.headers['Cache-Control']=='no-store'
            response=session.delete(base+'/api/v1/tasks/'+task['task_id'],timeout=3)
            assert response.json()['deleted']
            assert session.get(base+'/api/v1/tasks/'+task['task_id'],timeout=3).status_code==404
            report['delete_verified']=True
        finally:
            process.terminate();process.wait(timeout=20)
    (ROOT/'http_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'startup_seconds':report['startup_seconds'],'stages':stages,'warmup':health['warmup_metrics'],'timings':result['timings_ms']},ensure_ascii=True),flush=True)


if __name__=='__main__': main()
