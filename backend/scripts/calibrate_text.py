"""Small public development fixture, explicitly not a locked accuracy test."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.config import Settings
from app.pipeline import Pipeline
from app.imaging import decode_image,crop


if __name__=='__main__':
    p=Pipeline(Settings());p.load()
    records={}
    questions={
        'brows':'What is the shape of the eyebrows: straight, arched or angled? Answer with a short visible description.',
        'eyes':'Are the eyes open, narrowed or closed? Answer briefly about only the eyes.',
        'nose':'Describe the visible shape of the nose in one short sentence.',
        'mouth':'Are teeth visible? Are the lips parted or closed? Answer briefly about the mouth.',
        'contour':'Is the face shape oval, round, long or angular? Answer briefly about facial outline only.',
        'expression':'Is the person visibly smiling, frowning, or neutral? Answer briefly about only visible facial expression.'}
    for name,path in [('astronaut',p.settings.weights/'warmup.jpg'),('grace',p.settings.weights/'fixtures/grace_hopper.jpg')]:
        image=decode_image(path.read_bytes(),p.settings).bgr
        face,_=crop(image,p.detect(image)[0]['bbox_px'])
        records[name]={key:p.llm.generate(face,prompt,64,20) for key,prompt in questions.items()}
        print(name,json.dumps(records[name],ensure_ascii=True),flush=True)
    (p.settings.weights.parent/'text_development.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    p.close()
