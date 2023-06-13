


class Unlimiformer:

    def __init__(self):
        pass

    def break_into(self):
        pass

    def pre_eval_hook(self):
        pass

    def pre_train_hook(self):
        pass

    def inject_hooks(self, model):
        pass

    def inject_training_hooks(self, model):
        pass

    def inject_hooks_for_unaffected_layers(self, model, decoder_layers_to_run):
        pass

    def

class UnlimiLongformer(Unlimiformer):

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def create_key_value(self):
        pass

    def process_key_value(self):
        pass

    def process_query(self):
        pass

    def attention_layer_to_capture(self, layer_begin, layer_end):
        return [
            [layer.encoder_attn.k_proj, layer.encoder_attn.v_proj]
            for layer in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]

    def attention_op_to_run(self, layer_begin, layer_end):
        return [
            layer.encoder_attn.q_proj
            for layer in self.model.base_model.decoder.layers[layer_begin:layer_end]
        ]

    def attention_layer_to_run(self, layer_begin, layer_end):
        return self.model.base_model.decoder.layers[layer_begin:layer_end]

    def self_attention(self, decoder_layer):
        return decoder_layer.self_attn

    def cross_attention(self, decoder_layer):
        return decoder_layer.encoder_attn

    def window_size(self):
        pass

    def create_decoder_layer_args(self):
        pass

