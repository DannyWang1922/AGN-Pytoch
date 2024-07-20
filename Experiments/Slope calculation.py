import math

# def calculate_slope(batch_size, decay_rate, decay_steps):
#     total_step_in_one_epoch = 14041 // batch_size
#     num_decay = total_step_in_one_epoch / decay_steps
#     num_epoch_of_once_decay = 1 / num_decay
#     slope = math.exp(math.log(decay_rate) / num_epoch_of_once_decay)
#     print(f"{num_epoch_of_once_decay:.2f} epochs, lr decay {decay_rate}")
#     print("Slop: ", slope)
#
#
# calculate_slope(32, 0.85, 1250)


import math


def calculate_decay_steps(batch_size, decay_rate, slope):
    total_step_in_one_epoch = 14041 // batch_size
    num_epoch_of_once_decay = math.log(decay_rate) / math.log(slope)
    num_decay = 1 / num_epoch_of_once_decay
    decay_steps = total_step_in_one_epoch / num_decay
    print(f"{num_epoch_of_once_decay:.2f} epochs, lr decay {decay_rate}")
    print("Decay Steps: ", decay_steps)
    return decay_steps


# Example usage
calculate_decay_steps(32, 0.85, math.exp(math.log(0.85) / (14041 // 32 / 1250)))
