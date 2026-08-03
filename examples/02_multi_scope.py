import goggles as gg

# Each console handler needs its own bus-wide name; define them once here.
HANDLER1_NAME = "examples.multi_scope.console.1"
HANDLER2_NAME = "examples.multi_scope.console.2"
HANDLER3_NAME = "examples.multi_scope.console.3"

# In this example, we set up a handlers associated
# to different scopes.
handler1 = gg.ConsoleHandler(name=HANDLER1_NAME, level=gg.INFO)
gg.attach(handler1, scopes=["global", "namespace.scope1"])

handler2 = gg.ConsoleHandler(name=HANDLER2_NAME, level=gg.INFO)
gg.attach(handler2, scopes=["global", "namespace.scope2"])

# We need to get separate loggers for each scope
logger_scope1 = gg.get_logger("examples.basic.scope1", scope="namespace.scope1")
logger_scope2 = gg.get_logger("examples.basic.scope2", scope="namespace.scope2")
logger_scope2.bind(
    scope="namespace.scope2"
)  # You can also bind the scope after creation
logger_global = gg.get_logger("examples.basic.global", scope="global")

# Now we can log messages to different scopes, so that only the interested
# handlers will process them.
logger_scope1.info(f"This will be logged only by {handler1.name}")
logger_scope2.info(f"This will be logged only by {handler2.name}")
logger_global.info("This will be logged by both handlers.")

# The same result can be achieved using namespaces,
# which are indicated by dot notation.
handler3 = gg.ConsoleHandler(name=HANDLER3_NAME, level=gg.INFO)
gg.attach(handler3, scopes=["namespace"])
logger_scope1.info(
    f"This will be logged by {handler1.name} and {handler3.name}"
)
logger_scope2.info(
    f"This will be logged by {handler2.name} and {handler3.name}"
)

gg.finish()
