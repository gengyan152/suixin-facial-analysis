"""Run final tests and actual HTTP smoke serially, keeping explicit artifacts."""
from pathlib import Path
import os,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]


if __name__=='__main__':
    os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUTF8='1')
    with (ROOT/'final-tests.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','pytest','-q','--live','--basetemp','tests/.tmp-final','--junitxml=final-test-results.xml'],
                              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    print((ROOT/'final-tests.log').read_text(encoding='utf-8'),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    subprocess.run([sys.executable,'scripts/http_smoke.py'],cwd=ROOT,check=True)
