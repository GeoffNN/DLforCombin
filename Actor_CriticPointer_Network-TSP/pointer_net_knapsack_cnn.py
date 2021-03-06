import tensorflow as tf
import numpy as np
import tsp_env

def attention_mask(W_ref, W_q, v, enc_outputs, query, already_played_actions=None,
                   already_played_penalty=1e6, use_terminal_symbol=False):
    # import pdb; pdb.set_trace()
    with tf.variable_scope("attention_mask"):
        u_i0s = tf.einsum('kl,itl->itk', W_ref, enc_outputs)
        u_i1s = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
        if use_terminal_symbol:
            already_played_penalty_arr = already_played_penalty * np.ones((1, enc_outputs.get_shape()[1].value))
            already_played_penalty_arr[0, -1] = 0
            u_is = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s)) - \
                   tf.constant(already_played_penalty_arr, dtype=tf.float32) * already_played_actions
        else:
            u_is = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s)) - \
                   already_played_penalty * already_played_actions
        return u_is, tf.nn.softmax(u_is)

def makeCNNLayer(inputs, filters, dilation, residual=False, kernel_size=2):
    gate = tf.layers.conv1d(inputs = inputs,
                          filters = filters,
                          kernel_size = kernel_size,
                          dilation_rate = dilation,
                          padding="SAME",
                          activation=tf.sigmoid,
                          trainable = True)
    fil = tf.layers.conv1d(inputs = inputs,
                          filters = filters,
                          kernel_size = kernel_size,
                          dilation_rate = dilation,
                          padding="SAME",
                          activation=tf.tanh,
                          trainable = True)
    
    out = gate * fil
    
    if residual:
        return inputs + out
    else:
        return out
    
def pointer_network(enc_inputs, decoder_targets,
                    hidden_size=128, embedding_size=128,
                    max_time_steps=10, input_size=2,
                    batch_size=128,
                    initialization_stddev=0.1,
                    use_terminal_symbol=False):
    # Embed inputs in larger dimensional tensors
    W_embed = tf.Variable(tf.random_normal([embedding_size, input_size],
                                           stddev=initialization_stddev))
    embedded_inputs = tf.einsum('kl,itl->itk', W_embed, enc_inputs)

    # Define encoder
    with tf.variable_scope("encoder"):
        conv = []
        conv.append(makeCNNLayer(embedded_inputs, filters=hidden_size/4, dilation=1))
        # make the other layers
        numDilationLayer = 3
        factors = [2, 4, 8]
        for layerNum in range(0, numDilationLayer):
            conv.append(makeCNNLayer(conv[-1], filters=hidden_size/4, dilation=factors[layerNum], residual=True))

        enc_outputs = tf.concat(conv, axis=2)
        # enc_outputs = conv[-1]

        # avg pooling to mimic lstm state
        enc_final_state_c = tf.reduce_mean(enc_outputs, axis=1)

        # for "LSTM output (h)" do tanh(c)
        enc_final_state = tf.nn.rnn_cell.LSTMStateTuple(c = enc_final_state_c, h = tf.nn.tanh(enc_final_state_c))
        
    # Define decoder
    with tf.variable_scope("decoder"):
        decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
        first_decoder_input = tf.tile(tf.Variable(tf.random_normal([1, embedding_size]),
                                                  name='first_decoder_input'), [batch_size, 1])
        # Define attention weights
        with tf.variable_scope("attention_weights", reuse=True):
            W_ref = tf.Variable(tf.random_normal([embedding_size, embedding_size],
                                                 stddev=initialization_stddev),
                                name='W_ref')
            W_q = tf.Variable(tf.random_normal([embedding_size, embedding_size],
                                               stddev=initialization_stddev),
                              name='W_q')
            v = tf.Variable(tf.random_normal([embedding_size], stddev=initialization_stddev),
                            name='v')

        # Training chain
        loss = 0
        decoder_input = first_decoder_input
        decoder_state = enc_final_state
        already_played_actions = tf.zeros(shape=[batch_size, max_time_steps], dtype=tf.float32)
        decoder_inputs = [decoder_input]
        for t in range(max_time_steps):
            dec_cell_output, decoder_state = decoder_cell(inputs=decoder_input,
                                                          state=decoder_state)
            attn_logits, _ = attention_mask(W_ref, W_q, v, enc_outputs, dec_cell_output,
                                            already_played_actions=already_played_actions,
                                            already_played_penalty=1e6, use_terminal_symbol=use_terminal_symbol)
            loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets[:, t],
                                                                                            depth=max_time_steps),
                                                                          logits=attn_logits))
            loss_summary_sy = tf.summary.scalar('training_loss', loss)

            # Teacher forcing of the next input
            decoder_input = tf.einsum('itk,it->ik', embedded_inputs,
                                      tf.one_hot(decoder_targets[:, t], depth=max_time_steps))
            decoder_inputs.append(decoder_input)
            already_played_actions += tf.one_hot(decoder_targets[:, t], depth=max_time_steps)

        # Inference chain
        decoder_input = first_decoder_input
        decoder_state = enc_final_state
        decoder_outputs = []
        attn_masks = []
        already_played_actions = tf.zeros(shape=[batch_size, max_time_steps], dtype=tf.float32)
        for t in range(max_time_steps):
            dec_cell_output, decoder_state = decoder_cell(inputs=decoder_input,
                                                          state=decoder_state)
            _, attn_mask = attention_mask(W_ref, W_q, v, enc_outputs, dec_cell_output,
                                          already_played_actions=already_played_actions,
                                          already_played_penalty=1e6,
                                          use_terminal_symbol=use_terminal_symbol)
            decoder_outputs.append(tf.argmax(attn_mask, axis=1))
            decoder_input = tf.einsum('itk,it->ik', embedded_inputs, attn_mask)

            already_played_actions += tf.one_hot(decoder_outputs[-1],
                                                 depth=max_time_steps)
            attn_masks.append(attn_mask)
    return loss, loss_summary_sy, decoder_outputs, attn_masks
