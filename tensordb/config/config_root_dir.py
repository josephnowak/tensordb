# -*- coding: utf-8 -*-
"""
Root configuration file for Pypeline
"""

import os


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'tests')
TEST_DIR_TENSOR_CLIENT = os.path.join(TEST_DIR, 'data', 'test_tensor_client')
TEST_DIR_S3 = os.path.join(TEST_DIR, 'data', 'test_s3')
TEST_DIR_ZARR = os.path.join(TEST_DIR, 'data', 'test_zarr')
TEST_DIR_JSON = os.path.join(TEST_DIR, 'data', 'test_json_storage')
TEST_DIR_CACHED_TENSOR = os.path.join(TEST_DIR, 'data', 'test_cached_tensor')
