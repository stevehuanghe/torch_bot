
from torchbot.pipeline.engine import TorchEngine


if __name__ == "__main__":
    engine = TorchEngine(exp_name="example")
    engine.start()