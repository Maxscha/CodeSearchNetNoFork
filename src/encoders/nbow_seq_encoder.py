from typing import Dict, Any

import tensorflow as tf

from .masked_seq_encoder import MaskedSeqEncoder

def pool_sequence_embedding(pool_mode: str,
                            sequence_token_embeddings: tf.Tensor,
                            sequence_lengths: tf.Tensor,
                            sequence_token_masks: tf.Tensor) -> tf.Tensor:
    """
    Takes a batch of sequences of token embeddings and applies a pooling function,
    returning one representation for each sequence.

    Args:
        pool_mode: The pooling mode, one of "mean", "max", "weighted_mean". For
         the latter, a weight network is introduced that computes a score (from [0,1])
         for each token, and embeddings are weighted by that score when computing
         the mean.
        sequence_token_embeddings: A float32 tensor of shape [B, T, D], where B is the
         batch dimension, T is the maximal number of tokens per sequence, and D is
         the embedding size.
        sequence_lengths: An int32 tensor of shape [B].
        sequence_token_masks: A float32 tensor of shape [B, T] with 0/1 values used
         for masking out unused entries in sequence_embeddings.
    Returns:
        A tensor of shape [B, D], containing the pooled representation for each
        sequence.
    """
    print(pool_mode)
    if pool_mode == 'mean':
        seq_token_embeddings_masked = \
            sequence_token_embeddings * tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x D
        seq_token_embeddings_sum = tf.reduce_sum(seq_token_embeddings_masked, axis=1)  # B x D
        sequence_lengths = tf.expand_dims(tf.cast(sequence_lengths, dtype=tf.float32), axis=-1)  # B x 1
        return seq_token_embeddings_sum / sequence_lengths
    elif pool_mode == 'max':
        sequence_token_masks = -BIG_NUMBER * (1 - sequence_token_masks)  # B x T
        sequence_token_masks = tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x 1
        return tf.reduce_max(sequence_token_embeddings + sequence_token_masks, axis=1)
    elif pool_mode == 'weighted_mean':
        token_weights = tf.layers.dense(sequence_token_embeddings,
                                        units=1,
                                        activation=tf.sigmoid,
                                        use_bias=False)  # B x T x 1
        token_weights *= tf.expand_dims(sequence_token_masks, axis=-1)  # B x T x 1
        seq_embedding_weighted_sum = tf.reduce_sum(sequence_token_embeddings * token_weights, axis=1)  # B x D
        return seq_embedding_weighted_sum / (tf.reduce_sum(token_weights, axis=1) + 1e-8)  # B x D
    else:
        raise ValueError("Unknown sequence pool mode '%s'!" % pool_mode)



class NBoWEncoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = { 'nbow_pool_mode': 'weighted_mean',
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    def embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
        """
        Creates embedding layer that is in common between many encoders.

        Args:
            token_inp:  2D tensor that is of shape (batch size, sequence length)

        Returns:
            3D tensor of shape (batch size, sequence length, embedding dimension)
        """

        print("test embedding")
        token_embeddings = tf.get_variable(name='token_embeddings',
                                           initializer=tf.glorot_uniform_initializer(),
                                           shape=[len(self.metadata['token_vocab']),
                                                  self.get_hyper('token_embedding_size')],
                                           )
        self.__embeddings = token_embeddings

        token_embeddings = tf.nn.dropout(token_embeddings,
                                         keep_prob=self.placeholders['dropout_keep_rate'])

        return tf.nn.embedding_lookup(params=token_embeddings, ids=token_inp)

    @property
    def output_representation_size(self):
        return self.get_hyper('token_embedding_size')

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.variable_scope("nbow_encoder"):
            self._make_placeholders()

            seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
            seq_token_mask = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(seq_token_mask, axis=1)  # B
            return pool_sequence_embedding(self.get_hyper('nbow_pool_mode').lower(),
                                           sequence_token_embeddings=seq_tokens_embeddings,
                                           sequence_lengths=seq_token_lengths,
                                           sequence_token_masks=seq_token_mask)
