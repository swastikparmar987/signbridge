import os
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from scripts.scale_msasl_acquisition import (
    classify_youtube_error,
    get_cookie_options,
    ManifestManager,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestAcquisitionPipeline(unittest.TestCase):
    def test_classify_youtube_error(self):
        # 1. Unavailable videos
        status, _ = classify_youtube_error("ERROR: [youtube] 1AyT77LqJzQ: This video is unavailable")
        self.assertEqual(status, "unavailable")

        status, _ = classify_youtube_error("ERROR: [youtube] 9FdHlMOnVjg: Private video")
        self.assertEqual(status, "unavailable")

        status, _ = classify_youtube_error("ERROR: [youtube] Eq6SnaimpzQ: This video is not available")
        self.assertEqual(status, "unavailable")

        # 2. Bot challenge / sign-in requirements
        status, _ = classify_youtube_error("ERROR: [youtube] UXetwN_cI5A: Sign in to confirm you’re not a bot. Use --cookies-from-browser")
        self.assertEqual(status, "bot_challenge")

        status, _ = classify_youtube_error("ERROR: [youtube] TnJQtTYVTtg: Please sign in. Use --cookies-from-browser or --cookies")
        self.assertEqual(status, "bot_challenge")

        # 3. Transient network errors
        status, _ = classify_youtube_error("ERROR: unable to download video data: HTTP Error 429: Too Many Requests")
        self.assertEqual(status, "transient_error")

        status, _ = classify_youtube_error("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        self.assertEqual(status, "transient_error")

    def test_cookie_options_safety(self):
        # Default with no cookies passed or set
        opts = get_cookie_options()
        self.assertNotIn("cookiefile", opts)
        self.assertNotIn("cookiesfrombrowser", opts)

        # Explicit browser cookie
        opts_b = get_cookie_options(cookies_from_browser="chrome")
        self.assertEqual(opts_b.get("cookiesfrombrowser"), ("chrome",))

    def test_manifest_manager_skipping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dl_path = tmp_path / "download_manifest.csv"
            prep_path = tmp_path / "preprocessing_manifest.csv"
            landmarks_dir = tmp_path / "landmarks"
            landmarks_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy landmark file for successful sample
            valid_lm_file = landmarks_dir / "msasl_train_00001_landmarks.npy"
            np.save(valid_lm_file, np.zeros((32, 126), dtype=np.float32))

            records = [
                # Sample 1: Success with landmark file -> Should SKIP (is_completed == True)
                {
                    "sample_id": "msasl_train_00001",
                    "url": "https://youtube.com/watch?v=aaa",
                    "download_status": "success",
                    "preprocessing_status": "success",
                    "landmark_path": "landmarks/msasl_train_00001_landmarks.npy",
                    "failure_reason": "",
                },
                # Sample 2: Genuinely Unavailable -> Should SKIP (is_completed == True)
                {
                    "sample_id": "msasl_train_00002",
                    "url": "https://youtube.com/watch?v=bbb",
                    "download_status": "unavailable",
                    "preprocessing_status": "skipped",
                    "landmark_path": "",
                    "failure_reason": "Private video",
                },
                # Sample 3: Bot challenge -> Should RETRY (is_completed == False)
                {
                    "sample_id": "msasl_train_00003",
                    "url": "https://youtube.com/watch?v=ccc",
                    "download_status": "bot_challenge",
                    "preprocessing_status": "skipped",
                    "landmark_path": "",
                    "failure_reason": "Sign in to confirm you’re not a bot",
                },
                # Sample 4: Transient error -> Should RETRY (is_completed == False)
                {
                    "sample_id": "msasl_train_00004",
                    "url": "https://youtube.com/watch?v=ddd",
                    "download_status": "transient_error",
                    "preprocessing_status": "skipped",
                    "landmark_path": "",
                    "failure_reason": "HTTP Error 429",
                },
                # Sample 5: Generic download_error -> Should RETRY (is_completed == False)
                {
                    "sample_id": "msasl_train_00005",
                    "url": "https://youtube.com/watch?v=eee",
                    "download_status": "download_error",
                    "preprocessing_status": "skipped",
                    "landmark_path": "",
                    "failure_reason": "Download error",
                },
            ]

            df = pd.DataFrame(records)
            df.to_csv(dl_path, index=False)
            df.to_csv(prep_path, index=False)

            logger_mock = unittest.mock.MagicMock()
            manager = ManifestManager(dl_path, prep_path, {}, logger_mock)

            self.assertTrue(manager.is_completed("msasl_train_00001", landmarks_dir))
            self.assertTrue(manager.is_completed("msasl_train_00002", landmarks_dir))
            self.assertFalse(manager.is_completed("msasl_train_00003", landmarks_dir))
            self.assertFalse(manager.is_completed("msasl_train_00004", landmarks_dir))
            self.assertFalse(manager.is_completed("msasl_train_00005", landmarks_dir))


if __name__ == "__main__":
    unittest.main()
