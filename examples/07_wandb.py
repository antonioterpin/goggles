import argparse
import goggles as gg


def main(args):
    gg.configure(enable_console=False, enable_jsonl=True, enable_wandb=True)

    with gg.run(
        "wandb-demo",
        log_dir=args.log_dir,
        project="goggles-examples",  # goes into metadata for W&B init in your impl
    ) as ctx:
        gg.scalar("demo/value", 42, step=0)
        gg.get_logger("examples.wandb").info("logged-to-wandb-too")
        wb = " (mirrored to W&B)" if (ctx.wandb is not None) else " (W&B not available)"
        print(f"done{wb}; ctx.wandb = {ctx.wandb}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="runs")
    main(p.parse_args())
