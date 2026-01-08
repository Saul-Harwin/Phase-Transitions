# 1. 2D Ising Model Simulation in R
# This code simulates the 2D Ising model using the Metropolis algorithm.

set.seed(42)

# Parameters

# Oringal values:
# N_SWEEPS <- 5000  
# N_SAMPLES <- 1000
# NUM_TEMPS <- 20

N_SWEEPS <- 5000
N_SAMPLES <- 5000
NUM_TEMPS <- 30

# Prompt the user for name of the simulation run
filename <- readline(prompt = "Enter file name for the simulation run: ")


# 2. Lattice Representation in R
# Spin configuration
L <- 50
N <- L * L

# Initialize random spins
spins <- matrix(sample(c(-1, 1), N, replace = TRUE), nrow = L, ncol = L)

# 3. Periodic Boundary Conditions
# In R, we handle periodic boundaries explicitly.
pbc <- function(i, L) {
    if (i < 1) return(L)
    if (i > L) return(1)
    return(i)
}

# 4. Local Energy Change ΔE
# For Metropolis updates, we need the energy change for flipping a single spin.
delta_energy <- function(spins, i, j, J = 1) {
    L <- nrow(spins)

    s <- spins[i, j]

    nn_sum <- spins[pbc(i+1, L), j] +
        spins[pbc(i-1, L), j] +
        spins[i, pbc(j+1, L)] +
        spins[i, pbc(j-1, L)]

    return(2 * J * s * nn_sum)
}


# 5. One Metropolis Sweep
# A sweep consists of L^2 attempted spin flips.
metropolis_sweep <- function(spins, beta, J = 1) {
    L <- nrow(spins)

    # Random indices for L^2 attempted flips
    i <- sample(1:L, L*L, replace = TRUE)
    j <- sample(1:L, L*L, replace = TRUE)
  
    for (k in seq_along(i)) {
        ii <- i[k]
        jj <- j[k]

        # Periodic boundary indices
        ip <- ifelse(ii == L, 1, ii + 1)
        im <- ifelse(ii == 1, L, ii - 1)
        jp <- ifelse(jj == L, 1, jj + 1)
        jm <- ifelse(jj == 1, L, jj - 1)

        s <- spins[ii, jj]

        # Sum over nearest neighbors
        nn_sum <- spins[ip, jj] + spins[im, jj] + spins[ii, jp] + spins[ii, jm]

        dE <- 2 * J * s * nn_sum

        if (dE <= 0 || runif(1) < exp(-beta * dE)) {
            spins[ii, jj] <- -s
        }
  }
  
    return(spins)
}


# 6. Thermalization
# Before collecting data, we must equilibrate.
thermalize <- function(spins, beta, n_sweeps = N_SWEEPS) {
    for (s in 1:n_sweeps) {
        spins <- metropolis_sweep(spins, beta)
    }
    spins
}

# 7. Sampling Configurations for ML
# This is the critical part for the paper:
#   we collect raw configurations, not observables.
sample_configs <- function(spins, beta, n_samples = N_SAMPLES, gap = 10) {
    L <- nrow(spins)
    configs <- matrix(0, nrow = n_samples, ncol = L*L)
  
    for (n in 1:n_samples) {
        for (g in 1:gap) {
            spins <- metropolis_sweep(spins, beta)
        }
        configs[n, ] <- as.vector(spins)
    }
  
    return(list(spins = spins, configs = configs))
}

# 8. Full Simulation Loop Over Temperature
temperatures <- seq(1.5, 3.5, length.out = NUM_TEMPS)
configs_all <- list()
temps_all <- c()

spins <- matrix(sample(c(-1, 1), N, replace = TRUE), L, L)

pb <- txtProgressBar(min = 0, max = length(temperatures), style = 3)

for (i in seq_along(temperatures)) {
    T <- temperatures[i]
    beta <- 1 / T

    spins <- thermalize(spins, beta)

    result <- sample_configs(spins, beta, n_samples = N_SAMPLES)

    configs_all[[as.character(T)]] <- result$configs
    temps_all <- c(temps_all, rep(T, N_SAMPLES))

    spins <- result$spins

    # Update progress bar
    setTxtProgressBar(pb, i)
}

close(pb)

# 9. Preparing Data for PCA / Autoencoders
X <- do.call(rbind, configs_all)

# Center data
X_centered <- scale(X, center = TRUE, scale = FALSE)


# 10. Sanity Checks (Physics First)
magnetization <- rowMeans(X)
plot(temps_all, abs(magnetization), pch = 16, cex = 0.4)

# Export from R
library(rhdf5)

h5createFile(paste0(filename, ".h5"))

h5write(X_centered, paste0(filename, ".h5"), "configs")
h5write(temps_all, paste0(filename, ".h5"), "temps")

# Optional metadata
h5write(L, paste0(filename, ".h5"), "L")
h5write("2D Ising model", paste0(filename, ".h5"), "model")