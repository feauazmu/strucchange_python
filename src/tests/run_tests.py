import pytest

from bld.project_paths import project_paths_join as ppj

pytest.main([ppj("IN_MODEL_CODE", "")])
