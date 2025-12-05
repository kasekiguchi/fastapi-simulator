from .base import FURUTA_PENDULUM

# モジュールを「クラスのように見せる」ためのエイリアス
def __call__(*args, **kwargs):
    return FURUTA_PENDULUM(*args, **kwargs)

# これが鍵。モジュールの __call__ を設定する。
import sys
sys.modules[__name__].__call__ = __call__

__all__ = ["FURUTA_PENDULUM"]
