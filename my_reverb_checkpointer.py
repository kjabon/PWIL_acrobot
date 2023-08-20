from reverb.platform.checkpointers_lib import DefaultCheckpointer
import tempfile

# class ReverbCheckpointer(DefaultCheckpointer):
#   """Stores and loads checkpoints from a temporary directory."""
#
#   def __init__(self, path, fallback_path):
#     super().__init__(path=path,fallback_checkpoint_path=fallback_path)

def make_temp_dir_reverb_checkpointer(fallbackPath):
  class TempDirCheckpointerFromPath(DefaultCheckpointer):
    """Stores and loads checkpoints from a temporary directory."""

    def __init__(self):
      super().__init__(tempfile.mkdtemp(), fallback_checkpoint_path=fallbackPath)
  return TempDirCheckpointerFromPath


def make_reverb_checkpointer(path, fallback_path):
  class ReverbCheckpointer(DefaultCheckpointer):
    """Stores and loads checkpoints from a temporary directory."""

    def __init__(self):
      super().__init__(path=path, fallback_checkpoint_path=fallback_path)
  return ReverbCheckpointer
