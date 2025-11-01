import goggles as gg

# In this example, we set up a handlers associated
# to different scopes.
handler1 = gg.ConsoleHandler(name="examples.basic.console.1", level=gg.INFO)
gg.attach(handler1, scopes=["global", "scope1"])

handler2 = gg.ConsoleHandler(name="examples.basic.console.2", level=gg.INFO)
gg.attach(handler2, scopes=["global", "scope2"])

# We need to get separate loggers for each scope
logger_scope1 = gg.get_logger("examples.basic.scope1", scope="scope1")
logger_scope2 = gg.get_logger("examples.basic.scope2")
logger_scope2.bind(scope="scope2")  # You can also bind the scope after creation
logger_global = gg.get_logger("examples.basic.global", scope="global")

# Now we can log messages to different scopes, so that only the interested
# handlers will process them.
logger_scope1.info(f"This will be logged only by {handler1.name}")
logger_scope2.info(f"This will be logged only by {handler2.name}")
logger_global.info("This will be logged by both handlers.")
