"""One after_tool trial with every raw line from the claude CLI saved."""
import asyncio, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], "raw", "1"]
import petd.brain.claude_cli as cli                                   # noqa: E402

RAW = open(Path(__file__).parent / "raw_stream.jsonl", "w")


class Tee:
    @staticmethod
    def loads(line):
        RAW.write(line.decode() if isinstance(line, bytes) else line); RAW.flush()
        return json.loads(line)

    dumps = staticmethod(json.dumps)


cli.json = Tee
import runpy                                                          # noqa: E402
mod = runpy.run_path(str(Path(__file__).parent / "after_tool.py"), run_name="not_main")
mod["REQUESTS"][:] = ["Go back to your dock, please."]
asyncio.run(mod["main"]("raw", 1))
