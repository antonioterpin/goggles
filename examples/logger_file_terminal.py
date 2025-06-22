"""Example of logging to both file and terminal using Goggles."""

from goggles import Goggles, Severity

print("\n-- Test 1: Read default config --")
cfg = Goggles.get_config()
print(cfg)

print("\n-- Test 2: Logging to file and terminal, level=DEBUG --")
# {timestamp} will be replaced with current time
Goggles.set_config(
    name="test-{timestamp}", to_file=True, to_terminal=True, level=Severity.DEBUG
)
Goggles.debug("debug msg")
Goggles.info("info msg")

# Inspect log file
log_file = cfg["file_path"]
print(f"Log file {log_file} content:")
with open(log_file, "r") as f:
    print(f.read())

print("\n-- Test 3: Terminal only, WARNING level --")
Goggles.set_config(
    name="same-log-now", level=Severity.WARNING, to_file=False, to_terminal=True
)
print(Goggles.get_config())
Goggles.info("should NOT appear")
Goggles.warning("warning appears")

print("\n-- Test 4: Both outputs, INFO level --")
Goggles.set_config(
    name="same-log-now", level=Severity.INFO, to_file=True, to_terminal=True
)
print(Goggles.get_config())
Goggles.debug("debug omitted")
Goggles.info("info appears")

# Check cumulative log
cfg = Goggles.get_config()
print(f"Accumulated log file {cfg['file_path']}:")
with open(cfg["file_path"], "r") as f:
    print(f.read())
