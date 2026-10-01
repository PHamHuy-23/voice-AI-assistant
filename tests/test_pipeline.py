import os
import unittest
import numpy as np
import torch

from src.utils.config import load_yaml, set_seed
from src.data.preprocessing import AudioPreprocessor
from src.features.audio_features import FeatureExtractor
from src.models.td_resnet import TDResNetEmbeddingNet
from src.models.prototypical_network import PrototypicalNetwork
from src.models.distances import compute_euclidean_distance


class TestFewShotKWS(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.model_cfg = load_yaml("configs/model/td_resnet.yaml")
        self.prep_cfg = load_yaml("configs/preprocessing/normalized.yaml")
        self.proto_cfg = load_yaml("configs/model/prototypical.yaml")

    def test_audio_preprocessing_shapes(self):
        preprocessor = AudioPreprocessor(self.prep_cfg)
        # Create dummy stereo audio (2 channels, 24000 samples at 24kHz)
        dummy_audio = np.random.randn(2, 24000).astype(np.float32)
        processed, sr, meta = preprocessor.process(dummy_audio, orig_sr=24000)

        # Must be mono and exactly 16000 samples at 16kHz
        self.assertEqual(sr, 16000)
        self.assertEqual(processed.shape, (1, 16000))
        self.assertIn("to_mono", meta["steps_applied"])

    def test_feature_extraction(self):
        extractor = FeatureExtractor(self.model_cfg)
        dummy_mono = np.random.randn(1, 16000).astype(np.float32)

        mel = extractor.compute_mel_spectrogram(dummy_mono)
        mfcc = extractor.compute_mfcc(dummy_mono)
        tensor = extractor.extract_model_input(dummy_mono)

        self.assertEqual(mel.shape[0], 64)  # 64 Mel bands
        self.assertEqual(mfcc.shape[0], 40) # 40 MFCC coefficients
        self.assertEqual(tensor.ndim, 4)    # (1, 1, 64, T)
        self.assertEqual(tensor.shape[1], 1)
        self.assertEqual(tensor.shape[2], 64)

    def test_model_forward_and_distances(self):
        backbone = TDResNetEmbeddingNet(self.model_cfg)
        proto_net = PrototypicalNetwork(backbone, self.proto_cfg)

        # Batch of 5 support samples, 5 query samples
        dummy_input = torch.randn(5, 1, 64, 101)
        emb = backbone(dummy_input)
        self.assertEqual(emb.shape, (5, 128))

        # Check L2 norm
        norms = torch.norm(emb, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

        # Test distance
        x = torch.randn(10, 128)
        y = torch.randn(5, 128)
        dist = compute_euclidean_distance(x, y)
        self.assertEqual(dist.shape, (10, 5))
        self.assertTrue((dist >= 0).all())


if __name__ == "__main__":
    unittest.main()
