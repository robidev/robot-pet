"""
Local HTTP API on 127.0.0.1:8765: status, recent events, say, hear
(inject text), stop, snapshot, and a small debug dashboard at /.

The brain's MCP shim will call tools through here too (PLAN.md step D2).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from ..app import App

log = logging.getLogger(__name__)


class TextBody(BaseModel):
    text: str


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    return obj


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
        return [{"type": type(e).__name__, **to_jsonable(e)} for e in reversed(recent)]

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

    @api.get("/faces/current")
    async def faces_current() -> dict:
        if not pet.face:
            raise HTTPException(503, "face disabled")
        return to_jsonable(await pet.face.current_faces())

    return api


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
:root{--bg:#f6f7f9;--fg:#1d2127;--muted:#667;--card:#fff;--line:#dde1e6;--accent:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#131518;--fg:#e6e8eb;--muted:#9aa;--card:#1c1f23;--line:#2c3036}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,sans-serif;padding:16px;max-width:1100px;margin:auto}
h1{font-size:20px;margin:0 0 12px}.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 8px}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font-size:12px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
#stop{background:var(--accent);color:#fff;border:0;font-weight:600;padding:10px 20px}
input{font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);flex:1;min-width:0}
.row{display:flex;gap:6px;margin-bottom:8px}img{max-width:100%;border-radius:6px}
#events{max-height:420px;overflow:auto}
</style></head><body>
<div class="row" style="justify-content:space-between;align-items:center"><h1>{{NAME}}</h1><button id="stop">STOP</button></div>
<div class="grid">
<div class="card"><h2>Status</h2><pre id="status">…</pre></div>
<div class="card"><h2>Talk</h2>
<div class="row"><input id="say" placeholder="Make it say…"><button onclick="post('/say','say')">Say</button></div>
<div class="row"><input id="hear" placeholder="Pretend it heard…"><button onclick="post('/hear','hear')">Hear</button></div>
<h2>Camera</h2><button onclick="snap()">Snapshot</button><div><img id="img" alt=""></div></div>
<div class="card" style="grid-column:1/-1"><h2>Events</h2><pre id="events">…</pre></div>
</div>
<script>
async function post(url,id){const el=document.getElementById(id);if(!el.value)return;
await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:el.value})});el.value='';}
document.getElementById('stop').onclick=()=>fetch('/stop',{method:'POST'});
function snap(){document.getElementById('img').src='/snapshot?'+Date.now();}
function fmt(e){const t=new Date(e.t*1000).toLocaleTimeString();const {type,t:_,...rest}=e;
return t+'  '+type+'  '+JSON.stringify(rest);}
async function tick(){try{
const s=await (await fetch('/status')).json();document.getElementById('status').textContent=JSON.stringify(s,null,1);
const ev=await (await fetch('/events?n=80')).json();
document.getElementById('events').textContent=ev.filter(e=>e.type!=='VacuumStateChanged'||e.changed.some(c=>c!=='pose')).map(fmt).join('\\n');
}catch(e){}setTimeout(tick,1000);}
tick();
</script></body></html>
"""
