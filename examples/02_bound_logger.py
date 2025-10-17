import argparse
import goggles as gg


def main(args):
    gg.configure(enable_console=True, enable_file=True)

    with gg.run("bind", log_dir=args.log_dir, enable_jsonl=True) as ctx:
        log = gg.get_logger("examples.bind", exp="demo").bind(stage="train")
        log.info("start", step=0)
        # Chain more context without mutating the original adapter
        eval_log = log.bind(stage="eval")
        eval_log.info("evaluate", step=100, split="val")

        print(f"jsonl at: {ctx.log_dir}/events.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
