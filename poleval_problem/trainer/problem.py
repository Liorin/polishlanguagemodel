import os
import tensorflow as tf
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import generator_utils
from collections import Counter

OOV_TOKEN = "<UNK>"

def _create_vocab(vocab_file_path):
    with open("data/small.txt") as f:
        cnts = Counter([ word for line in f for word in line.strip().split() ])
    list_of_tokens = [ key for key in cnts.keys() if cnts[key] >= 3 ]
    encoder = text_encoder.TokenTextEncoder(None, vocab_list = [OOV_TOKEN] + list_of_tokens, replace_oov = OOV_TOKEN)
    encoder.store_to_file(vocab_file_path)
    return encoder



@registry.register_problem
class PolishLanguageProblem(text_problems.Text2SelfProblem):
    """Polish language model from NKJP - National Polish Language Corpus."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
        return OOV_TOKEN

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def vocab_filename(self):
        return "small.vocab"

    def get_or_create_vocab(self, data_dir, tmp_dir, force_get=False):
        vocab_path = os.path.join(data_dir, self.vocab_filename)
        if not tf.gfile.Exists(vocab_path):
            return _create_vocab(vocab_path)
        #else
        return text_encoder.TokenTextEncoder(vocab_path)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        _create_vocab(os.path.join(data_dir, self.vocab_filename))
        with tf.gfile.Open('data/small.txt', "r") as f:
            for line in f:
                yield {
                    "targets": line.strip()
                }


# Smaller than the typical translate model, and with more regularization
@registry.register_hparams
def transformer_polish_language_poleval():
  hparams = transformer.transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 32
  hparams.filter_size = 32
  hparams.num_heads = 2
  hparams.attention_dropout = 0.6
  hparams.layer_prepostprocess_dropout = 0.6
  hparams.learning_rate = 0.05
  return hparams

# hyperparameter tuning ranges
@registry.register_ranged_hparams
def transformer_poetry_range(rhp):
  rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
  rhp.set_int("num_hidden_layers", 2, 4)
  rhp.set_discrete("hidden_size", [128, 256, 512])
  rhp.set_float("attention_dropout", 0.4, 0.7)
