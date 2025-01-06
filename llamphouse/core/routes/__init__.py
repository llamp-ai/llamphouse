from .assistant import router as assistant_router
from .threads import router as threads_router
from .message import router as message_router
from .run import router as run_router

all_routes = [assistant_router, threads_router, message_router, run_router]
