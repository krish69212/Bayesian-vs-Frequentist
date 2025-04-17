import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

st.set_page_config(page_title="Bayesian vs Frequentist", layout="centered")
st.title("Bayesian vs Frequentist: Coin Toss Example")

# User input
st.sidebar.header("Simulation Settings")
n_flips = st.sidebar.slider("Number of coin flips", 1, 100, 10)
n_heads = st.sidebar.slider("Number of heads observed", 0, n_flips, 7)

st.header("Frequentist Approach")

# Frequentist estimate
freq_p = n_heads / n_flips
conf_interval = [freq_p - 1.96 * np.sqrt(freq_p * (1 - freq_p) / n_flips),
                 freq_p + 1.96 * np.sqrt(freq_p * (1 - freq_p) / n_flips)]

st.write(f"Estimated Probability of Heads: **{freq_p:.2f}**")
st.write(f"95% Confidence Interval: **[{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]**")

st.latex(r"""
\hat{p} = \frac{\text{Number of Heads}}{\text{Number of Flips}} = \frac{%d}{%d} = %.2f
""" % (n_heads, n_flips, freq_p))

st.latex(r"""
CI = \hat{p} \pm 1.96 \times \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
""")

# Plot Frequentist Point Estimate
fig, ax = plt.subplots()
ax.axvline(freq_p, color='blue', label='Point Estimate')
ax.axvspan(conf_interval[0], conf_interval[1], color='blue', alpha=0.2, label='95% CI')
ax.set_xlim(0, 1)
ax.set_title('Frequentist Estimate')
ax.set_xlabel('Probability of Heads')
ax.set_ylabel('Density')
ax.legend()
st.pyplot(fig)

st.header("Bayesian Approach")

# Prior
prior_alpha = 1
prior_beta = 1

# Posterior
posterior_alpha = prior_alpha + n_heads
posterior_beta = prior_beta + (n_flips - n_heads)
x = np.linspace(0, 1, 1000)
posterior = beta.pdf(x, posterior_alpha, posterior_beta)

st.write(f"Posterior Distribution: Beta({posterior_alpha}, {posterior_beta})")

st.latex(r"""
\text{Posterior} \sim \text{Beta}(\alpha + x, \beta + n - x)
""")

st.latex(r"""
\text{Posterior} \sim \text{Beta}(%d + %d, %d + %d) = \text{Beta}(%d, %d)
""" % (prior_alpha, n_heads, prior_beta, n_flips - n_heads, posterior_alpha, posterior_beta))

# Plot Posterior
fig2, ax2 = plt.subplots()
ax2.plot(x, posterior, label='Posterior', color='red')
ax2.set_title('Bayesian Posterior Distribution')
ax2.set_xlabel('Probability of Heads')
ax2.set_ylabel('Density')
ax2.legend()
st.pyplot(fig2)

st.markdown("---")
st.markdown("This app was created by **Krish Parmar** to demonstrate the key differences between Bayesian and Frequentist statistical thinking using a simple coin toss experiment.")