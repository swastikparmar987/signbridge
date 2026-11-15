import unittest
import numpy as np

from signbridge.continuous import (
    ContinuousCSLRSessionManager,
    ContinuousLatticeDecoder,
    KinematicStrokeSegmenter,
    resample_sequence_aware,
)


def _make_synthetic_sign_stroke(n_frames: int, shift_x: float = 0.0) -> np.ndarray:
    """Create a realistic moving right-hand sign stroke with handshape articulation."""
    seq = np.zeros((n_frames, 42, 3), dtype=np.float32)
    base_hand = np.zeros((21, 3), dtype=np.float32)
    for j in range(21):
        finger = (j - 1) // 4 if j > 0 else 0
        joint = (j - 1) % 4 + 1 if j > 0 else 0
        base_hand[j, 0] = 0.50 + shift_x + (finger - 2) * 0.025
        base_hand[j, 1] = 0.55 - joint * 0.035
        base_hand[j, 2] = -0.02 * joint

    t = np.linspace(0.0, np.pi, n_frames, dtype=np.float32)
    # Strong palm arc + finger curl wave in mid-stroke, tapering at endpoints
    vel_env = np.sin(t)[:, None]
    for i in range(n_frames):
        frame_h = base_hand.copy()
        frame_h[:, 0] += 0.14 * np.sin(t[i] * 1.2)
        frame_h[:, 1] -= 0.12 * vel_env[i, 0]
        # Curl index/middle tips mid-sign
        frame_h[8, 1] += 0.05 * np.sin(2.0 * t[i])
        frame_h[12, 1] += 0.05 * np.sin(2.0 * t[i])
        seq[i, 21:42] = frame_h
    return seq


def _make_static_rest(n_frames: int, x: float = 0.5, y: float = 0.72) -> np.ndarray:
    seq = np.zeros((n_frames, 42, 3), dtype=np.float32)
    hand = np.zeros((21, 3), dtype=np.float32)
    for j in range(21):
        hand[j, 0] = x + (j % 5) * 0.015
        hand[j, 1] = y - (j // 5) * 0.025
    seq[:, 21:42] = hand[None, :, :]
    return seq


class TestContinuousCSLR(unittest.TestCase):
    def test_resample_sequence_aware_preserves_zero_hand_slot(self):
        # First 10 frames: only right hand (21:42); rest left hand all zeros
        raw = _make_synthetic_sign_stroke(15)
        res = resample_sequence_aware(raw, target_len=32)
        self.assertEqual(res.shape, (32, 42, 3))
        self.assertTrue(np.allclose(res[:, :21], 0.0))
        self.assertGreater(np.abs(res[:, 21:42]).sum(), 1.0)

    def test_segmenter_isolates_two_signs_and_pause_finalizes(self):
        seg = KinematicStrokeSegmenter(pause_finalize_ms=900.0)
        sign1 = _make_synthetic_sign_stroke(20, shift_x=-0.05)
        dip = _make_static_rest(4, x=0.5, y=0.52)
        sign2 = _make_synthetic_sign_stroke(20, shift_x=0.08)
        rest = _make_static_rest(40, x=0.5, y=0.75)

        stream = np.concatenate([sign1, dip, sign2, rest], axis=0)
        triggers = []
        pauses = []
        t = 0.0
        for i in range(len(stream)):
            wins, pause = seg.push_frame(stream[i], ts_ms=t)
            for w in wins:
                triggers.append((w.trigger, w.start_idx, w.end_idx, round(w.motion_score, 4)))
            if pause:
                pauses.append(i)
            t += 33.333

        self.assertTrue(any(tr[0] in ("dip_boundary", "pause_boundary") for tr in triggers), f"triggers={triggers}")
        self.assertTrue(len(pauses) > 0, "Expected pause_finalized after 40 rest frames (1.3s)")

    def test_session_manager_commits_glosses_without_duplicating_hold(self):
        mgr = ContinuousCSLRSessionManager()
        sign1 = _make_synthetic_sign_stroke(20, shift_x=-0.10)
        pause_mid = _make_static_rest(8, x=0.45, y=0.55)
        sign2 = _make_synthetic_sign_stroke(20, shift_x=0.12)
        rest_end = _make_static_rest(45, x=0.50, y=0.75)
        stream = np.concatenate([sign1, pause_mid, sign2, rest_end], axis=0)

        def fake_predict(seq_32: np.ndarray, top_k: int = 5):
            # Look at centroid x of active hand to simulate two distinct signs vs rest
            active = seq_32[:, 21:42, :]
            mean_x = float(active[:, :, 0].mean())
            disp = float(np.linalg.norm(np.diff(active[:, 0, :], axis=0), axis=-1).mean())
            if disp < 0.0025:
                return {"gloss": "NO_SIGN", "confidence": 0.85, "is_no_sign": True, "ood_prob": 0.05, "top_k": []}
            if mean_x < 0.51:
                return {
                    "gloss": "HELLO",
                    "confidence": 0.78,
                    "is_no_sign": False,
                    "ood_prob": 0.96,
                    "top_k": [{"gloss": "HELLO", "confidence": 0.78}, {"gloss": "HI", "confidence": 0.14}],
                }
            return {
                "gloss": "FRIEND",
                "confidence": 0.74,
                "is_no_sign": False,
                "ood_prob": 0.95,
                "top_k": [{"gloss": "FRIEND", "confidence": 0.74}, {"gloss": "FAMILY", "confidence": 0.16}],
            }

        res_final = None
        for idx in range(0, len(stream), 6):
            chunk = stream[idx : idx + 6]
            ts = [(idx + k) * 33.333 for k in range(len(chunk))]
            out = mgr.push_frames("test_sess", chunk, fake_predict, timestamps_ms=ts)
            if out["pause_finalized"]:
                res_final = out["finalized_utterance"]
                break

        self.assertIsNotNone(res_final)
        self.assertEqual(res_final["glosses"], ["HELLO", "FRIEND"])
        self.assertEqual(len(res_final["hypotheses"]), 2)
        self.assertEqual(res_final["hypotheses"][0][1]["gloss"], "HI")

    def test_pause_separated_repeated_gloss_commits_twice(self):
        mgr = ContinuousCSLRSessionManager()
        s1 = _make_synthetic_sign_stroke(18, shift_x=0.0)
        mid_pause = _make_static_rest(14, x=0.5, y=0.6)
        s2 = _make_synthetic_sign_stroke(18, shift_x=0.0)
        end_pause = _make_static_rest(48, x=0.5, y=0.75)
        stream = np.concatenate([s1, mid_pause, s2, end_pause], axis=0)

        def yes_predict(seq_32: np.ndarray, top_k: int = 5):
            return {
                "gloss": "YES",
                "confidence": 0.72,
                "is_no_sign": False,
                "ood_prob": 0.96,
                "top_k": [{"gloss": "YES", "confidence": 0.72}],
            }

        res = None
        for idx in range(0, len(stream), 6):
            chunk = stream[idx : idx + 6]
            ts = [(idx + k) * 33.333 for k in range(len(chunk))]
            out = mgr.push_frames("rep_sess", chunk, yes_predict, timestamps_ms=ts)
            if out["pause_finalized"]:
                res = out["finalized_utterance"]
                break

        self.assertIsNotNone(res)
        self.assertEqual(res["glosses"], ["YES", "YES"])

    def test_short_subtle_stroke_emits_pause_boundary_with_natural_cadence(self):
        from signbridge.continuous import resample_natural_cadence

        seg = KinematicStrokeSegmenter()
        # Subtle 5-frame finger-curl sign followed by hand drop out of frame
        onset = _make_static_rest(3, x=0.5, y=0.6)
        short_sign = np.repeat(_make_static_rest(1, x=0.5, y=0.6), 6, axis=0)
        for i in range(6):
            # Subtle finger-tap articulation across fingertips (mimics WATER)
            for k in (4, 8, 12, 16, 20):
                short_sign[i, 21 + k, 1] += 0.055 * np.sin((i + 1) * np.pi / 6.0)
        zeros = np.zeros((36, 42, 3), dtype=np.float32)

        stream = np.concatenate([onset, short_sign, zeros], axis=0)
        boundaries = []
        paused = False
        t = 0.0
        for f in stream:
            wins, pf = seg.push_frame(f, ts_ms=t)
            for w in wins:
                boundaries.append(w)
            paused = paused or pf
            t += 33.333

        self.assertTrue(paused)
        self.assertTrue(
            any(w.trigger == "pause_boundary" for w in boundaries),
            f"Expected pause_boundary for 5-frame subtle sign, got {[w.trigger for w in boundaries]}",
        )
        # Ensure natural cadence zero-pads short strokes instead of over-stretching 6x
        padded = resample_natural_cadence(short_sign, target_len=32)
        nz_frames = int((np.abs(padded).sum(axis=(1, 2)) > 1e-4).sum())
        self.assertLessEqual(nz_frames, 12)

    def test_decoder_refines_same_stroke_live_tail_without_duplicate(self):
        dec = ContinuousLatticeDecoder()
        t1 = dec.observe_window(
            0.0,
            280.0,
            "live_tail",
            {"predicted_gloss": "FIREMAN1", "confidence": 0.64, "is_no_sign": False, "ood_prob": 0.99, "top_k": [{"gloss": "FIREMAN1", "confidence": 0.64}]},
            stroke_id=1,
        )
        self.assertIsNotNone(t1)
        t2 = dec.observe_window(
            0.0,
            420.0,
            "live_tail",
            {"predicted_gloss": "HELLO", "confidence": 0.78, "is_no_sign": False, "ood_prob": 0.99, "top_k": [{"gloss": "HELLO", "confidence": 0.78}]},
            stroke_id=1,
        )
        self.assertIsNone(t2)
        self.assertEqual(dec.get_glosses(), ["HELLO"])


if __name__ == "__main__":
    unittest.main()
