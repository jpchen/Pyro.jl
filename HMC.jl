using Distributions
using LinearAlgebra: dot
using Random
using Test

# Hamiltonian Monte Carlo algorithm
function hmc(logpdf, grad_logpdf, initial_position, num_samples, step_size, num_steps)
    samples = []
    current_position = initial_position
    current_logpdf = logpdf(current_position)
    current_grad_logpdf = grad_logpdf(current_position)

    for i in 1:num_samples
        position = current_position
        momentum = randn(length(position))  # Random Gaussian momentum
        proposed_momentum = momentum
        proposed_position = position

        # Leapfrog integration
        proposed_momentum -= 0.5 * step_size * current_grad_logpdf
        for j in 1:num_steps
            proposed_position += step_size * proposed_momentum
            if j != num_steps
                proposed_momentum -= step_size * grad_logpdf(proposed_position)
            end
        end
        proposed_momentum -= 0.5 * step_size * grad_logpdf(proposed_position)

        # Metropolis acceptance step
        proposed_logpdf = logpdf(proposed_position)
        acceptance_ratio = exp(proposed_logpdf - current_logpdf + 
                               0.5 * dot(momentum, momentum) - 
                               0.5 * dot(proposed_momentum, proposed_momentum))
        if rand() < acceptance_ratio
            current_position = proposed_position
            current_logpdf = proposed_logpdf
            current_grad_logpdf = grad_logpdf(proposed_position)
        end

        push!(samples, current_position)
    end

    return samples
end

# Log probability density function and its gradient for a toy model
logpdf_normal(x) = logpdf(Normal(0, 1), x)
grad_logpdf_normal(x) = -x  # Gradient of log-pdf for standard normal distribution

# Test the HMC algorithm
initial_position = 0.0
num_samples = 1000
step_size = 0.1
num_steps = 10

samples = hmc(logpdf_normal, grad_logpdf_normal, initial_position, num_samples, step_size, num_steps)

print(samples)

@test length(samples) == num_samples
@test all(abs.(samples) .<= 3)
