import os
import json

from tensordb.backup_handlers import S3Handler
from tensordb.config.config_root_dir import TEST_DIR_S3


# TODO: Create an account in AWS and try to receive support from amazon

def get_default_s3_handler():
    return S3Handler(
        aws_access_key_id='',
        aws_secret_access_key='',
        region_name=''
    )


class TestS3Handler:
    data = {'test': 'test'}

    def _write_json(self):
        with open(os.path.join(TEST_DIR_S3, 'test.json'), mode='w') as json_file:
            json.dump(self.data, json_file)

    def test_upload_file(self):
        assert True
        # s3_handler = get_default_s3_handler()
        # self._write_json()
        # s3_handler.upload_file(
        #     bucket_name='',
        #     local_path=os.path.join(TEST_DIR_S3, 'test.json'),
        #     s3_path=os.path.join('test_s3', 'test.json')
        # )

    def test_download_file(self):
        assert True
        # self.test_upload_file()
        # s3_handler = get_default_s3_handler()
        # s3_handler.download_file(
        #     bucket_name='',
        #     local_path=os.path.join(TEST_DIR_S3, 'test.json'),
        #     s3_path=os.path.join('test_s3', 'test.json')
        # )
        # with open(os.path.join(TEST_DIR_S3, 'test.json'), mode='r') as json_file:
        #     data_s3 = json.load(json_file)
        #
        # assert TestS3Handler.data == data_s3

    def test_get_head_object(self):
        assert True
        # self.test_upload_file()
        # s3_handler = get_default_s3_handler()
        # head = s3_handler.get_head_object(
        #     bucket_name='',
        #     s3_path=os.path.join('test_s3', 'test.json')
        # )
        # last_modified_date = s3_handler.get_last_modified_date(
        #     bucket_name='',
        #     s3_path=os.path.join('test_s3', 'test.json')
        # )
        # etag = s3_handler.get_etag(
        #     bucket_name='',
        #     s3_path=os.path.join('test_s3', 'test.json')
        # )
        # assert last_modified_date.strftime('%Y-%m-%d') == head['LastModified'].strftime('%Y-%m-%d')


if __name__ == "__main__":
    test = TestS3Handler()
    test.test_upload_file()
    # test.test_download_file()
    # test.test_get_head_object()


