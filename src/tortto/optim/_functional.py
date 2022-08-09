from tortto import *

def sgd(params, d_p_list, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p + param.data * weight_decay
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = d_p
                momentum_buffer_list[i] = d_p
            else:
                buf *= momentum
                buf += d_p * (1 - dampening)
            if nesterov:
                d_p = d_p + buf * momentum
            else:
                d_p = buf

        param.data += d_p * -lr


def adam(params,  # list of tensor
         grads,  # list of ndarray
         exp_avgs,  # list of ndarray
         exp_avg_sqs,  # list of ndarray
         max_exp_avg_sqs,  # list of ndarray
         state_steps,
         amsgrad,
         beta1,
         beta2,
         lr,
         weight_decay,
         eps):

    if grads:
        xp = cp if grads[0].__class__ is cp_ndarray else np

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad + param.data * weight_decay

        # Decay the first and second moment running average coefficient
        exp_avg *= beta1
        exp_avg += (1 - beta1) * grad

        exp_avg_sq *= beta2
        exp_avg_sq += grad * grad.conj() * (1 - beta2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            xp.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (xp.sqrt(max_exp_avg_sqs[i]) / xp.sqrt(bias_correction2)) + eps
        else:
            denom = (xp.sqrt(exp_avg_sq) / xp.sqrt(bias_correction2)) + eps

        step_size = lr / bias_correction1
        param.data += -step_size * exp_avg / denom


def adamw(params,  # list of tensor
          grads,  # list of ndarray
          exp_avgs,  # list of ndarray
          exp_avg_sqs,  # list of ndarray
          max_exp_avg_sqs,  # list of ndarray
          state_steps,
          amsgrad,
          beta1,
          beta2,
          lr,
          weight_decay,
          eps):

    if grads:
        xp = cp if grads[0].__class__ is cp_ndarray else np

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # Perform stepweight decay
        param.data *= 1 - lr * weight_decay

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg *= beta1
        exp_avg += (1 - beta1) * grad

        exp_avg_sq *= beta2
        exp_avg_sq += grad * grad.conj() * (1 - beta2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            xp.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (xp.sqrt(max_exp_avg_sqs[i]) / xp.sqrt(bias_correction2)) + eps
        else:
            denom = (xp.sqrt(exp_avg_sq) / xp.sqrt(bias_correction2)) + eps

        step_size = lr / bias_correction1
        param.data += -step_size * exp_avg / denom