import torch

from app.core.models.nets.functions.pulser_function import PulserFunction


class HybridPulserFunction(PulserFunction):
    @staticmethod
    def forward(#ctx, 
                x_projected, mask, surrogate_model, device, backend_name, use_noise, backend_params, waves_size, phase_param_size, qubit_positions):
        #ctx.original_shape = x_projected.shape
        #ctx.surrogate_model = surrogate_model

        # vera simulazione quantistica
        real_out = PulserFunction.forward(
            #ctx, 
            x_projected=x_projected, 
            mask=mask,
            device=device, 
            backend_name=backend_name, 
            use_noise=use_noise, 
            backend_params=backend_params, 
            waves_size=waves_size, 
            phase_param_size=phase_param_size,
            qubit_positions=qubit_positions
        )
        
        # forward del surrogato per agganciarlo al grafo di calcolo di PyTorch
        batch_size, num_nodes, params_per_node = x_projected.shape
        x_flat = x_projected.reshape(batch_size, num_nodes * params_per_node)
        pos_flat = qubit_positions.reshape(batch_size, -1)
        surrogate_out = surrogate_model(x_flat, pos_flat)
        surrogate_out = surrogate_out.reshape(real_out.shape)

        # salviamo input appiattito
        # ctx.save_for_backward(x_flat)
        
        # STE: restituisci solo valore di pulser nel fw 
        # e aggiungiamo al grafo dei gradienti solo il fw del surrogato
        return real_out.detach() + surrogate_out - surrogate_out.detach()

    @staticmethod
    def backward(ctx, grad_output):
        surrogate_model = ctx.surrogate_model
        x_flat, = ctx.saved_tensors
        original_shape = ctx.original_shape
        
        # creiamo una copia dell'input appiattito che richiede
        #  il gradiente esplicitamente ed eseguiamo il modello surrogato
        x_flat_var = x_flat.detach().clone().requires_grad_(True)
        surrogate_out = surrogate_model(x_flat_var)
        
        # gradiente di surrogate_out rispetto a x_flat_var 
        # moltiplicandola per il grad_output proveniente dalla loss (Chain Rule)
        grad_flat = torch.autograd.grad(
            outputs=surrogate_out,
            inputs=x_flat_var,
            grad_outputs=grad_output,
            retain_graph=False
        )[0]
        
        # riportiamo il gradiente alla forma originaria [batch_size, num_nodes, params_per_node]
        grad_input = grad_flat.reshape(original_shape)
        
        return grad_input, None, None, None, None, None, None, None, None, None