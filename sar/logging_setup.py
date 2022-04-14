# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Union
import logging

logger = logging.getLogger('sar')
logger.addHandler(logging.NullHandler())
logger.propagate = False


def logging_setup(log_level: int, _rank: int = -1, _world_size: int = -1, log_file: Optional[str] = None):
    formatter = logging.Formatter(
        f'{_rank+1} / {_world_size} - %(name)s - %(levelname)s - %(message)s')

    handler: Union[logging.FileHandler, logging.StreamHandler]
    if log_file is not None:
        handler = logging.FileHandler(log_file, 'w')
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    logger.setLevel(log_level)
    logger.addHandler(handler)
