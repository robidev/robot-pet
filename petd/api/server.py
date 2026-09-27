"""
Local HTTP API on 127.0.0.1:8765: status, recent events, say, hear
(inject text), stop, snapshot, the tools, and a debug dashboard at / (G3):
the live map with places and people, who is in view and what recognition
is doing (with the latest face crops), the event log, and a tool console.

The brain's MCP shim calls its tools through here too (PLAN.md step D2).
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from ..jsonable import event_record, to_jsonable

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)


class TextBody(BaseModel):
    text: str


def create_app(pet: "App") -> FastAPI:
    api = FastAPI(title="petd", docs_url="/docs")

    @api.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD_HTML.replace("{{NAME}}", pet.cfg.pet.name)

    @api.get("/status")
    def status() -> dict:
        return {
            "name": pet.cfg.pet.name,
            "fake": pet.fake,
            "vacuum": to_jsonable(pet.vacuum.state) if pet.vacuum else None,
            "face": to_jsonable(pet.face.state) if pet.face else None,
            "faces_present": pet.face.presence.present if pet.face else None,
            "speaking": pet.speaker.speaking if pet.speaker else None,
            "stt_running": getattr(getattr(pet.stt, "process", None), "running", None),
        }

    @api.get("/events")
    def events(n: int = 50) -> list:
        recent = list(pet.bus.history)[-n:]
        return [event_record(e) for e in reversed(recent)]

    @api.post("/say")
    def say(body: TextBody) -> dict:
        if not pet.speaker:
            raise HTTPException(503, "speaker disabled")
        utt = pet.speaker.say(body.text)
        return {"utterance_id": utt.id}

    @api.post("/hear")
    def hear(body: TextBody) -> dict:
        pet.hear(body.text, source="api")
        return {"ok": True}

    @api.post("/stop")
    async def stop() -> dict:
        await pet.stop_everything("api")
        return {"ok": True}

    @api.get("/snapshot")
    async def snapshot() -> Response:
        if not pet.face:
            raise HTTPException(503, "face disabled")
        snap = await pet.face.snapshot()
        if snap is None:
            raise HTTPException(404, "no snapshot (fake face)")
        return Response(snap.jpeg, media_type="image/jpeg",
                        headers={"X-Capture-Timestamp": snap.utc or "",
                                 "X-Servo-Pan-Deg": str(snap.servo_pan_deg),
                                 "X-Servo-Tilt-Deg": str(snap.servo_tilt_deg)})

    @api.get("/tools")
    def tools() -> list:
        """The tool set, as the MCP shim and non-MCP backends see it."""
        if pet.tools is None:
            raise HTTPException(503, "no tool registry")
        return pet.tools.list()

    @api.post("/tool/{name}")
    async def call_tool(name: str, arguments: dict | None = None) -> dict:
        if pet.tools is None:
            raise HTTPException(503, "no tool registry")
        if name not in pet.tools:
            raise HTTPException(404, f"no such tool: {name}")
        result = await pet.tools.call(name, arguments or {})
        return {"text": result.text, "image_b64": result.image_b64,
                "mime": result.mime, "is_error": result.is_error}

    @api.get("/map.png")
    async def map_png(scale: int = 4) -> Response:
        """The live map: robot, dock, a go_to in progress, named places, where people were seen."""
        vacuum = pet.vacuum
        if vacuum is None or not vacuum.last_map:
            raise HTTPException(404, "no map yet (robot unreachable, or --fake)")
        markers = await map_markers(pet)
        from valetudo_client import render_map

        def draw() -> bytes:
            out = io.BytesIO()
            render_map(vacuum.last_map, scale=max(1, min(scale, 8)), markers=markers).save(out, "PNG")
            return out.getvalue()
        return Response(await asyncio.to_thread(draw), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @api.get("/people")
    def people() -> dict:
        """Who is in view, what recognition is doing, and the latest attempts."""
        now = time.time()
        present = [p.name for p in pet.people.present.values()] if pet.people else []
        rec = pet.recognizer
        names = {p.id: p.name for p in pet.db.people()} if pet.db is not None else {}
        tracks = []
        if rec is not None:
            for t in rec.tracks:
                last = t.samples[-1][1] if t.samples else None
                tracks.append({"centre": [round(c, 2) for c in t.centre],
                               "name": names.get(t.person_id) if t.person_id is not None else None,
                               "attempts": len(t.samples),
                               "last_best": names.get(last.best_id) if last else None,
                               "last_similarity": round(last.similarity, 2) if last else None,
                               "age_s": round(now - t.first_seen, 1)})
        attempts = []
        keeper = getattr(rec, "keeper", None)
        if keeper is not None:
            for a in reversed(keeper.recent_attempts(12)):
                attempts.append({"file": a.path.name, "clock": a.clock, "verdict": a.verdict,
                                 "best": a.best, "similarity": a.similarity})
        face = pet.face
        return {
            "present": present,
            "faces_in_view": pet.people.faces_in_view if pet.people else None,
            "presence": face.presence.present if face else None,
            "recognition": None if rec is None else {
                "on": True, "visiting": rec._visiting(), "tracks": tracks,
                "people_with_a_face": len(rec.centres)},
            "attempts": attempts,
        }

    @api.get("/person")
    async def person() -> dict:
        """Where each face the head sees now is, on the map (spatial/person.py, C5)."""
        return {"calibrated": pet.cfg.calibration.camera_calibrated,
                "people": [to_jsonable(e) | {"text": e.describe()} for e in await person_estimates(pet)]}

    @api.get("/faces/attempt/{name}")
    def face_attempt(name: str) -> Response:
        """One kept attempt's crop (runtime/faces/attempts/); only attempt file names."""
        from ..vision.kept import ATTEMPT_NAME
        keeper = getattr(pet.recognizer, "keeper", None)
        if keeper is None or not ATTEMPT_NAME.match(name):
            raise HTTPException(404, "no such attempt")
        path = keeper.attempts / name
        if not path.is_file():
            raise HTTPException(404, "no such attempt (rotated out?)")
        return Response(path.read_bytes(), media_type="image/jpeg")

    @api.get("/faces/current")
    async def faces_current() -> dict:
        if not pet.face:
            raise HTTPException(503, "face disabled")
        return to_jsonable(await pet.face.current_faces())

    return api


PLACE_COLOUR = (255, 140, 0)
PERSON_COLOUR = (140, 60, 200)
ESTIMATE_COLOUR = (220, 0, 140)
TARGET_COLOUR = (0, 170, 170)
_grid_cache: list = [None, None]            # [map json it was made from, Grid]


def current_grid(pet: "App"):
    """The map as a Grid (mapgeo), decoded once per map the vacuum adapter polled."""
    raw = pet.vacuum.last_map if pet.vacuum else None
    if raw is None:
        return None
    if _grid_cache[0] is not raw:
        from ..spatial.mapgeo import Grid
        _grid_cache[:] = [raw, Grid.from_valetudo(raw)]
    return _grid_cache[1]


async def person_estimates(pet: "App") -> list:
    """An estimate for each face the head sees now; the eye height of the one known person if alone."""
    if pet.face is None:
        return []
    from ..spatial.person import estimate
    frame = await pet.face.current_faces()
    if not frame.faces:
        return []
    face_z = None
    if pet.people is not None and len(frame.faces) == 1 and len(pet.people.present) == 1:
        face_z = next(iter(pet.people.present.values())).face_z_m
    pose = pet.vacuum.state.pose if pet.vacuum else None
    grid = await asyncio.to_thread(current_grid, pet)
    return [estimate(f, frame.pan_deg, frame.tilt_deg, pet.cfg.calibration, pet.cfg.person, pose, grid, face_z)
            for f in frame.faces]


async def map_markers(pet: "App") -> list:
    """Places and last sightings, stored in the reference frame, moved onto the current map."""
    db, frame = pet.db, pet.frame
    if db is None:
        return []
    stored = [(*db.place(name), name, PLACE_COLOUR) for name in db.places()]
    stored += [(p.last_seen_x, p.last_seen_y, f"{p.name} seen from here", PERSON_COLOUR)
               for p in db.people() if p.last_seen_x is not None]
    markers = []
    for x, y, label, colour in stored:
        spot = await frame.to_current(x, y) if frame is not None else (x, y)
        if spot is not None:
            markers.append((*spot, label, colour))
    try:
        for e in await person_estimates(pet):     # already in the current map's frame
            if e.xy is not None:
                markers.append((*e.xy, "person " + e.describe(), ESTIMATE_COLOUR))
            if e.target is not None:
                markers.append((*e.target, "approach target", TARGET_COLOUR))
    except Exception:  # noqa: BLE001 - the map draws without them
        log.debug("no person estimates for the map", exc_info=True)
    return markers


async def serve(pet: "App") -> None:
    import uvicorn
    config = uvicorn.Config(create_app(pet), host=pet.cfg.api.host, port=pet.cfg.api.port,
                            log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    # petd owns signal handling; don't let uvicorn install its own.
    server.install_signal_handlers = lambda: None
    log.info("local API on http://%s:%d", pet.cfg.api.host, pet.cfg.api.port)
    await server.serve()


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{NAME}} · petd</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d2127;--muted:#667;--card:#fff;--line:#dde1e6;--accent:#c2410c;--ok:#15803d;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#131518;--fg:#e6e8eb;--muted:#9aa;--card:#1c1f23;--line:#2c3036;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif;padding:16px;max-width:1280px;margin:auto}
h1{font-size:20px;margin:0}.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.wide{grid-column:1/-1}@media (min-width:700px){.span2{grid-column:span 2}}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 8px}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:12px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
#stop{background:var(--accent);color:#fff;border:0;font-weight:600;padding:10px 20px}
input,select,textarea{font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);min-width:0}
input[type=text],textarea{flex:1}textarea{width:100%;font-family:ui-monospace,monospace;font-size:12px}
.row{display:flex;gap:6px;margin-bottom:8px;align-items:center;flex-wrap:wrap}img{max-width:100%;border-radius:6px}
#events{max-height:460px;overflow:auto}.muted{color:var(--muted)}.ok{color:var(--ok)}.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{text-align:left;padding:2px 6px;border-bottom:1px solid var(--line)}
.thumbs{display:flex;gap:6px;flex-wrap:wrap}.thumbs figure{margin:0;width:84px;font-size:11px;text-align:center}
.thumbs img{width:84px;height:84px;display:block}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:0}dt{color:var(--muted)}dd{margin:0}
</style></head><body>
<div class="row" style="justify-content:space-between"><h1>{{NAME}}</h1><button id="stop">STOP</button></div>
<div class="grid">
<div class="card"><h2>Status</h2><dl id="summary"></dl>
<details><summary class="muted">raw</summary><pre id="status">…</pre></details></div>
<div class="card"><h2>Talk</h2>
<div class="row"><input type="text" id="say" placeholder="Make it say…"><button onclick="post('/say','say')">Say</button></div>
<div class="row"><input type="text" id="hear" placeholder="Pretend it heard…"><button onclick="post('/hear','hear')">Hear</button></div>
<h2>Camera</h2><button onclick="snap()">Snapshot</button><div><img id="img" alt=""></div></div>
<div class="card"><h2>People</h2><dl id="presence"></dl>
<table id="tracks"></table><h2 style="margin-top:10px">Latest attempts</h2><div class="thumbs" id="attempts"></div></div>
<div class="card span2"><h2>Map</h2><div class="row"><label><input type="checkbox" id="maplive" checked> refresh every 5 s</label>
<span class="muted">blue robot · green dock · red go_to · orange places · purple last seen · magenta a person now, cyan where it would stop</span></div>
<img id="map" alt=""><div class="muted" id="nomap" hidden>no map yet: the robot is unreachable, or this is --fake</div></div>
<div class="card wide"><h2>Events</h2>
<div class="row"><input type="text" id="filter" placeholder="filter by type or text">
<label><input type="checkbox" id="quiet" checked> hide face boxes and pose updates</label>
<button id="pause">Pause</button></div><pre id="events">…</pre></div>
<div class="card wide"><h2>Tools</h2>
<div class="row"><select id="tool"></select><button onclick="callTool()">Call</button></div>
<div class="muted" id="tooldesc"></div><textarea id="args" rows="4">{}</textarea>
<pre id="toolout"></pre><img id="toolimg" alt=""></div>
</div>
<script>
const $=id=>document.getElementById(id);
async function post(url,id){const el=$(id);if(!el.value)return;
await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:el.value})});el.value='';}
$('stop').onclick=()=>fetch('/stop',{method:'POST'});
function snap(){$('img').src='/snapshot?'+Date.now();}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'})[c]);}
function dl(el,pairs){el.innerHTML=pairs.map(([k,v])=>'<dt>'+esc(k)+'</dt><dd>'+v+'</dd>').join('');}
function flag(on,yes,no){return on?'<span class="ok">'+yes+'</span>':'<span class="warn">'+no+'</span>';}
function summarize(s){const v=s.vacuum||{},f=s.face||{},p=v.pose||{};
dl($('summary'),[['robot',v.reachable===false?'<span class="warn">unreachable</span>':esc((v.status||'?')+(v.battery_level!=null?' · '+v.battery_level+'%':''))],
['pose',p.x!=null?esc(Math.round(p.x)+', '+Math.round(p.y)+', '+Math.round(p.angle||0)+'°'):'–'],
['face',f.reachable===false?'<span class="warn">unreachable</span>':esc('head '+(f.servo_mode||'?')+' · eye '+(f.eye_mode||'?'))],
['hearing',flag(s.stt_running,'listening','not running')],['speaking',s.speaking?'yes':'no'],['mode',s.fake?'fake hardware':'real hardware']]);}
function people(p){dl($('presence'),[['in view',p.presence?(p.faces_in_view+' face(s)'):'nobody'],
['known here',p.present.length?esc(p.present.join(', ')):'–'],
['recognition',!p.recognition?'off':(p.recognition.visiting?'<span class="ok">looking</span>':'idle')+' · '+p.recognition.people_with_a_face+' people stored']]);
const t=(p.recognition&&p.recognition.tracks)||[];
$('tracks').innerHTML=t.length?'<tr><th>where</th><th>name</th><th>attempts</th><th>last best</th><th>for</th></tr>'+t.map(r=>
'<tr><td>'+r.centre.join(', ')+'</td><td>'+esc(r.name||'unknown')+'</td><td>'+r.attempts+'</td><td>'+esc((r.last_best||'–')+' '+(r.last_similarity??''))+'</td><td>'+r.age_s+' s</td></tr>').join(''):'';
const key=p.attempts.map(a=>a.file).join();if($('attempts').dataset.key===key)return;$('attempts').dataset.key=key;
$('attempts').innerHTML=p.attempts.map(a=>'<figure><img src="/faces/attempt/'+encodeURIComponent(a.file)+'" alt="">'+
esc(a.clock)+'<br>'+esc(a.verdict||'?')+' '+a.similarity.toFixed(2)+'</figure>').join('')||'<span class="muted">none kept</span>';}
const NOISY=new Set(['FacesChanged','MotionDetected']);let paused=false;
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause';};
function fmt(e){const t=new Date(e.t*1000).toLocaleTimeString();const {type,t:_,...rest}=e;return t+'  '+type+'  '+JSON.stringify(rest);}
function showEvents(ev){if(paused)return;const f=$('filter').value.toLowerCase(),q=$('quiet').checked;
$('events').textContent=ev.filter(e=>!(q&&(NOISY.has(e.type)||(e.type==='VacuumStateChanged'&&e.changed.every(c=>c==='pose')))))
.map(fmt).filter(l=>!f||l.toLowerCase().includes(f)).join('\\n');}
let tools=[];
async function loadTools(){try{tools=await (await fetch('/tools')).json();}catch(e){return;}
$('tool').innerHTML=tools.map(t=>'<option>'+esc(t.name)+'</option>').join('');$('tool').onchange=pickTool;pickTool();}
function pickTool(){const t=tools.find(t=>t.name===$('tool').value);if(!t)return;$('tooldesc').textContent=t.description;
const props=(t.schema&&t.schema.properties)||{};const a={};
for(const [k,v] of Object.entries(props))a[k]=v.default??(v.type==='number'||v.type==='integer'?0:v.type==='boolean'?true:'');
$('args').value=JSON.stringify(a,null,1);}
async function callTool(){let args;try{args=JSON.parse($('args').value||'{}');}catch(e){$('toolout').textContent='arguments: '+e;return;}
$('toolout').textContent='…';$('toolimg').removeAttribute('src');
const r=await fetch('/tool/'+encodeURIComponent($('tool').value),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(args)});
const j=await r.json();$('toolout').textContent=(j.is_error?'ERROR: ':'')+(j.text??JSON.stringify(j));
if(j.image_b64)$('toolimg').src='data:'+(j.mime||'image/jpeg')+';base64,'+j.image_b64;}
function refreshMap(){if(!$('maplive').checked)return;const img=new Image();
img.onload=()=>{$('map').src=img.src;$('map').hidden=false;$('nomap').hidden=true;};
img.onerror=()=>{$('map').hidden=true;$('nomap').hidden=false;};img.src='/map.png?'+Date.now();}
async function tick(){try{
const s=await (await fetch('/status')).json();$('status').textContent=JSON.stringify(s,null,1);summarize(s);
people(await (await fetch('/people')).json());
showEvents(await (await fetch('/events?n=150')).json());
}catch(e){}setTimeout(tick,1000);}
tick();loadTools();refreshMap();setInterval(refreshMap,5000);
</script></body></html>
"""
