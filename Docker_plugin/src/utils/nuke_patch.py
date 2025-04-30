import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('nuke_samurai')

# Create a mock ProgressTask class
class ProgressTask:
    def __init__(self, name, total_steps=None):
        self.name = name
        self.total_steps = total_steps
        self.current_step = 0
        self._cancelled = False
        logger.info(f"Starting progress task: {name}")

    def setProgress(self, step):
        self.current_step = step
        if self.total_steps:
            progress = (step / self.total_steps) * 100
            logger.info(f"Progress: {progress:.1f}% - {self.name}")
        else:
            logger.info(f"Progress: {step} - {self.name}")

    def setMessage(self, message):
        logger.info(f"Progress message: {message}")

    def isCancelled(self):
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        logger.info(f"Progress task cancelled: {self.name}")

# Create a mock nuke module
class MockNuke:
    @staticmethod
    def tprint(message):
        logger.info(message)
    
    @staticmethod
    def ProgressTask(name, total_steps=None):
        return ProgressTask(name, total_steps)
    
    @staticmethod
    def executeInMainThread(func, args=None):
        if args:
            func(*args)
        else:
            func()

# Patch nuke.tprint
sys.modules['nuke'] = MockNuke() 