# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#讓 Python 將 sam2 目錄視為一個可導入的套件
#針對 SAM2 專案的特定功能：一導入就初始化 Hydra
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")
