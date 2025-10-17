import argparse, json
from pathlib import Path
import goggles as gg


def read_jsonl(path: Path):
    if not path.exists():
        print(f"(no {path.name} generated)")
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main(args):
    gg.configure(enable_console=False, enable_jsonl=True)  # JSONL on by default

    with gg.run("jsonl", log_dir=args.log_dir) as ctx:
        log = gg.get_logger("examples.jsonl", job="reader")
        log.info("hello-jsonl", epoch=0)
        log.warning("careful", detail="something-to-note")

        jsonl = Path(ctx.log_dir) / "events.jsonl"
        rows = read_jsonl(jsonl)
        print(f"read {len(rows)} rows from {jsonl}")
        # Show how bound vs per-call appear when using our adapter:
        for r in rows[-2:]:
            flat = {**r, **r.get("_g_bound", {}), **r.get("_g_extra", {})}
            print({k: flat.get(k) for k in ("msg", "job", "epoch", "detail")})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
